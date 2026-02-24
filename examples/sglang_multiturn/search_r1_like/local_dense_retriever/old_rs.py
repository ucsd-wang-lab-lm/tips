import json
import os
import warnings
from typing import List, Dict, Optional
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor

import faiss
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModel
from tqdm import tqdm
import datasets

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# Thread-local storage keeps a dedicated tokenizer per thread
_thread_local = threading.local()

def get_thread_local_tokenizer(model_path: str = "gpt2"):
    """
    Return the tokenizer instance bound to the current thread.

    A new tokenizer is created when none exists or when the requested model path changes.
    """
    # Check whether a new tokenizer needs to be created
    if (not hasattr(_thread_local, 'tokenizer') or 
        not hasattr(_thread_local, 'tokenizer_model_path') or
        _thread_local.tokenizer_model_path != model_path or
        _thread_local.tokenizer is None):
        try:
            from transformers import AutoTokenizer
            _thread_local.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            _thread_local.tokenizer_model_path = model_path
            if _thread_local.tokenizer.pad_token is None:
                _thread_local.tokenizer.pad_token = _thread_local.tokenizer.eos_token
        except Exception as e:
            print(f"Warning: Failed to load tokenizer {model_path}: {e}")
            _thread_local.tokenizer = None
            _thread_local.tokenizer_model_path = None
    return _thread_local.tokenizer

def truncate_document_content_fast(content: str, max_tokens: int = 500, max_chars: int = 2000) -> str:
    """
    Efficiently truncate document content.

    The logic uses a character budget to approximate the desired token limit.

    Args:
        content: The document text to truncate.
        max_tokens: Target token count (kept for compatibility; max_chars is primary).
        max_chars: Hard character cap (roughly equal to 500 tokens).

    Returns:
        The truncated content.
    """
    content_len = len(content)
    
    # Simplify by enforcing a pure character limit
    if content_len <= max_chars:
        return content
    
    # Cap content by characters to stay within max_chars
    target_chars = max_chars
    
    # Reserve space for the truncation marker
    marker_reserve = 80  # Reserve 80 characters for the truncation marker (enough for marker info)
    available_chars = target_chars - marker_reserve  # Characters available for actual content
    
    if available_chars <= 10:
        # If too little room remains, return a simple truncated prefix
        return content[:target_chars-10] + "(truncated)"
    
    # Allocate characters proportionally: 70% head, 30% tail
    head_chars = int(available_chars * 0.7)
    tail_chars = available_chars - head_chars
    
    # Ensure both head and tail keep at least one character
    head_chars = max(1, head_chars)
    tail_chars = max(1, tail_chars)
    
    # Adjust if the total exceeds the available character budget
    if head_chars + tail_chars > available_chars:
        head_chars = available_chars - 1
        tail_chars = 1
    
    # Grab the initial head and tail slices
    head_text = content[:head_chars]
    tail_text = content[-tail_chars:] if tail_chars > 0 else ""
    
    # Smart split: scan backward for the nearest break (space, newline, punctuation)
    if head_chars > 0:
        for i in range(min(50, head_chars)):  # Limit the search window to 50 characters
            check_pos = head_chars - 1 - i
            if check_pos >= 0 and content[check_pos] in [' ', '\n', '.', '!', '?', ';', ':', ',']:
                head_text = content[:check_pos + 1].rstrip()
                break
    
    # Smart split: scan forward for the nearest break
    if tail_chars > 0:
        tail_start = content_len - tail_chars
        for i in range(min(50, tail_chars)):  # Limit the search window to 50 characters
            check_pos = tail_start + i
            if check_pos < content_len and content[check_pos] in [' ', '\n', '.', '!', '?', ';', ':', ',']:
                tail_text = content[check_pos:].lstrip()
                break
    
    # Determine how many characters were removed
    actual_truncated_chars = content_len - len(head_text) - len(tail_text)
    estimated_tokens = actual_truncated_chars // 4  # Roughly assume 4 characters per token
    
    # Build the truncation marker
    marker = f"\n...(truncated ~{actual_truncated_chars} chars ≈ {estimated_tokens} tokens)...\n"
    
    # Assemble the final string
    final_result = head_text.rstrip() + marker + tail_text.lstrip()
    
    # Final safety check to ensure we do not exceed target_chars
    if len(final_result) > target_chars:
        # If we overshoot, fall back to a simpler marker
        # Recompute using a lean marker
        simple_marker = f"\n(truncated {actual_truncated_chars})\n"
        max_content_chars = target_chars - len(simple_marker)
        
        if max_content_chars > 10:
            # Redistribute the remaining character budget
            new_head_chars = int(max_content_chars * 0.7)
            new_tail_chars = max_content_chars - new_head_chars
            
            # Hard truncate without additional smart splitting to keep logic simple
            new_head_text = content[:new_head_chars].rstrip()
            new_tail_text = content[-new_tail_chars:].lstrip()
            
            final_result = new_head_text + simple_marker + new_tail_text
        else:
            # Extreme fallback: keep only the beginning when space is too tight
            final_result = content[:target_chars-15] + "\n(truncated)\n"

    return final_result

def truncate_document_content(content: str, max_tokens: int = 500, tokenizer_model_path: str = "gpt2") -> str:
    """
    Truncate document content to the specified token budget.

    Uses a smart strategy that keeps both the prefix and suffix of the text.
    """
    # Fetch the thread-local tokenizer
    tokenizer = get_thread_local_tokenizer(tokenizer_model_path)
    
    if tokenizer is not None:
        # Thread-local tokenizers avoid locking
        tokens = tokenizer.encode(content, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            return content
        
        # Split the title from the body
        lines = content.split('\n')
        if len(lines) < 2:
            # No clear title/body separation, do a direct truncation
            # Reserve tokens for the truncation marker first
            marker_text = f"\n...(truncated {len(tokens)} tokens)...\n"
            marker_tokens = len(tokenizer.encode(marker_text, add_special_tokens=False))
            available_tokens = max_tokens - marker_tokens
            
            if available_tokens <= 10:
                # If too few tokens remain, keep only the start
                safe_tokens = max(10, max_tokens - 20)
                safe_text = tokenizer.decode(tokens[:safe_tokens], skip_special_tokens=True)
                return safe_text + "\n(truncated)"
            
            # Allocate the remaining tokens: 70% head, 30% tail
            head_tokens = int(available_tokens * 0.7)
            tail_tokens = available_tokens - head_tokens
            
            # Make sure we stay in bounds
            if head_tokens + tail_tokens >= len(tokens):
                return content
                
            truncated_tokens = len(tokens) - head_tokens - tail_tokens
            
            head_text = tokenizer.decode(tokens[:head_tokens], skip_special_tokens=True)
            tail_text = tokenizer.decode(tokens[-tail_tokens:], skip_special_tokens=True)
            
            return (
                head_text + 
                f"\n...(truncated {truncated_tokens} tokens)...\n" + 
                tail_text
            )
        
        title = lines[0]
        body = '\n'.join(lines[1:])
        
        # Reserve tokens for the title and the truncation marker
        title_tokens = tokenizer.encode(title, add_special_tokens=False)
        truncate_marker_tokens = tokenizer.encode("\n...(truncated)...\n", add_special_tokens=False)
        available_tokens = max_tokens - len(title_tokens) - len(truncate_marker_tokens)
        
        if available_tokens <= 10:
            return title + "\n(content too long, truncated)"
        
        # Truncate the body
        body_tokens = tokenizer.encode(body, add_special_tokens=False)
        if len(body_tokens) <= available_tokens:
            return content
        
        # Allocate remaining tokens: 70% to the head, 30% to the tail
        head_tokens = int(available_tokens * 0.7)
        tail_tokens = available_tokens - head_tokens
        
        # Ensure we stay within bounds
        if head_tokens + tail_tokens >= len(body_tokens):
            return content
            
        truncated_tokens = len(body_tokens) - head_tokens - tail_tokens
        
        head_text = tokenizer.decode(body_tokens[:head_tokens], skip_special_tokens=True)
        tail_text = tokenizer.decode(body_tokens[-tail_tokens:], skip_special_tokens=True)
        
        truncated_body = (
            head_text + 
            f"\n...(truncated {truncated_tokens} tokens)...\n" + 
            tail_text
        )
        
        return title + '\n' + truncated_body
    else:
        # Fallback: estimate tokens conservatively at roughly 3 characters/token
        estimated_tokens = len(content) / 3
        
        if estimated_tokens <= max_tokens:
            return content
        
        # Split title and body
        lines = content.split('\n')
        if len(lines) < 2:
            # Without a clear title separation, perform a simple truncation
            max_chars = int(max_tokens * 3)
            if len(content) <= max_chars:
                return content
            
            head_chars = max_chars // 2
            tail_chars = max_chars // 3
            truncated_chars = len(content) - head_chars - tail_chars
            
            return (
                content[:head_chars] + 
                f"\n...(truncated {truncated_chars} characters)...\n" + 
                content[-tail_chars:]
            )
        
        title = lines[0]
        body = '\n'.join(lines[1:])
        
        # Reserve capacity for the title
        title_tokens = len(title) / 3
        available_tokens = max_tokens - title_tokens - 10  # Keep a small token buffer
        
        if available_tokens <= 0:
            return title + "\n(content too long, truncated)"
        
        # Truncate the body text
        body_chars = int(available_tokens * 3)
        if len(body) <= body_chars:
            return content
        
        head_chars = body_chars // 2
        tail_chars = body_chars // 3
        truncated_chars = len(body) - head_chars - tail_chars
        
        truncated_body = (
            body[:head_chars] + 
            f"\n...(truncated {truncated_chars} characters)...\n" + 
            body[-tail_chars:]
        )
        
        return title + '\n' + truncated_body

def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset(
        'json', 
        data_files=corpus_path,
        split="train",
        num_proc=4
    )
    return corpus

def read_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_docs(corpus, doc_idxs):
    results = [corpus[int(idx)] for idx in doc_idxs]
    return results

def load_model(model_path: str, use_fp16: bool = False):
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    if use_fp16: 
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer

def pooling(
    pooler_output,
    last_hidden_state,
    attention_mask = None,
    pooling_method = "mean"
):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")

class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16

        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)
        self.model.eval()
        
        # Save the tokenizer configuration so thread-local copies can be created
        self._tokenizer_config = {
            'model_path': model_path,
            'use_fast': True
        }

    @torch.no_grad()
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        # processing query for different encoders
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        if "bge" in self.model_name.lower():
            if is_query:
                query_list = [f"Represent this sentence for searching relevant passages: {query}" for query in query_list]

        # Retrieve the tokenizer for this thread
        thread_tokenizer = get_thread_local_tokenizer(self.model_path)
        if thread_tokenizer is None:
            # Fallback to the original tokenizer if the thread-local instance fails
            thread_tokenizer = self.tokenizer
            
        inputs = thread_tokenizer(query_list,
                                max_length=self.max_length,
                                padding=True,
                                truncation=True,
                                return_tensors="pt"
                                )
        inputs = {k: v.cuda() for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros(
                (inputs['input_ids'].shape[0], 1), dtype=torch.long
            ).to(inputs['input_ids'].device)
            output = self.model(
                **inputs, decoder_input_ids=decoder_input_ids, return_dict=True
            )
            query_emb = output.last_hidden_state[:, 0, :]
        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(output.pooler_output,
                                output.last_hidden_state,
                                inputs['attention_mask'],
                                self.pooling_method)
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")
        
        del inputs, output
        torch.cuda.empty_cache()

        return query_emb

class BaseRetriever:
    def __init__(self, config):
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk
        
        self.index_path = config.index_path
        self.corpus_path = config.corpus_path
        self.max_content_tokens = getattr(config, 'max_content_tokens', 500)

    def _search(self, query: str, num: int, return_score: bool):
        raise NotImplementedError

    def _batch_search(self, query_list: List[str], num: int, return_score: bool):
        raise NotImplementedError

    def search(self, query: str, num: int = None, return_score: bool = False):
        return self._search(query, num, return_score)
    
    def batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        return self._batch_search(query_list, num, return_score)

class BM25Retriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        from pyserini.search.lucene import LuceneSearcher
        self.searcher = LuceneSearcher(self.index_path)
        self.contain_doc = self._check_contain_doc()
        if not self.contain_doc:
            self.corpus = load_corpus(self.corpus_path)
        self.max_process_num = 8
        
        # No tokenizer needed here; rely on the thread-local helper
    
    def _check_contain_doc(self):
        return self.searcher.doc(0).raw() is not None

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        hits = self.searcher.search(query, num)
        if len(hits) < 1:
            if return_score:
                return [], []
            else:
                return []
        scores = [hit.score for hit in hits]
        if len(hits) < num:
            warnings.warn('Not enough documents retrieved!')
        else:
            hits = hits[:num]

        if self.contain_doc:
            all_contents = [
                json.loads(self.searcher.doc(hit.docid).raw())['contents'] 
                for hit in hits
            ]
            results = [
                {
                    'title': content.split("\n")[0].strip("\""),
                    'text': "\n".join(content.split("\n")[1:]),
                    'contents': truncate_document_content_fast(content, max_tokens=500, max_chars=2000)
                } 
                for content in all_contents
            ]
        else:
            raw_results = load_docs(self.corpus, [hit.docid for hit in hits])
            results = []
            for doc in raw_results:
                if 'contents' in doc:
                    doc['contents'] = truncate_document_content_fast(doc['contents'], max_tokens=500, max_chars=2000)
                results.append(doc)

        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        results = []
        scores = []
        for query in query_list:
            item_result, item_score = self._search(query, num, True)
            results.append(item_result)
            scores.append(item_score)
        if return_score:
            return results, scores
        else:
            return results

class DenseRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        self.index = faiss.read_index(self.index_path)
        if config.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)

        self.corpus = load_corpus(self.corpus_path)
        self.encoder = Encoder(
            model_name = self.retrieval_method,
            model_path = config.retrieval_model_path,
            pooling_method = config.retrieval_pooling_method,
            max_length = config.retrieval_query_max_length,
            use_fp16 = config.retrieval_use_fp16
        )
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        query_emb = self.encoder.encode(query)
        scores, idxs = self.index.search(query_emb, k=num)
        idxs = idxs[0]
        scores = scores[0]
        raw_results = load_docs(self.corpus, idxs)
        
        # Apply content truncation
        results = []
        for doc in raw_results:
            if 'contents' in doc:
                doc['contents'] = truncate_document_content_fast(doc['contents'], max_tokens=500, max_chars=2000)
            results.append(doc)
        
        if return_score:
            return results, scores.tolist()
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk
        
        results = []
        scores = []
        for start_idx in tqdm(range(0, len(query_list), self.batch_size), desc='Retrieval process: '):
            query_batch = query_list[start_idx:start_idx + self.batch_size]
            batch_emb = self.encoder.encode(query_batch)
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()

            # load_docs is not vectorized, but is a python list approach
            flat_idxs = sum(batch_idxs, [])
            raw_batch_results = load_docs(self.corpus, flat_idxs)
            
            # Apply content truncation in batches for efficiency
            truncated_batch_results = []
            contents_to_truncate = []
            doc_indices = []
            
            for i, doc in enumerate(raw_batch_results):
                if 'contents' in doc:
                    contents_to_truncate.append(doc['contents'])
                    doc_indices.append(i)
                truncated_batch_results.append(doc)
            
            # Batch truncation pass
            if contents_to_truncate:
                for idx, content in enumerate(contents_to_truncate):
                    doc_idx = doc_indices[idx]
                    truncated_batch_results[doc_idx]['contents'] = truncate_document_content_fast(
                        content, max_tokens=500, max_chars=2000
                    )
            
            # chunk them back
            batch_results = [truncated_batch_results[i*num : (i+1)*num] for i in range(len(batch_idxs))]
            
            results.extend(batch_results)
            scores.extend(batch_scores)
            
            del batch_emb, batch_scores, batch_idxs, query_batch, flat_idxs, batch_results
            torch.cuda.empty_cache()
            
        if return_score:
            return results, scores
        else:
            return results

def get_retriever(config):
    if config.retrieval_method == "bm25":
        return BM25Retriever(config)
    else:
        return DenseRetriever(config)


#####################################
# FastAPI server below
#####################################

class Config:
    """
    Minimal config class (simulating your argparse) 
    Replace this with your real arguments or load them dynamically.
    """
    def __init__(
        self, 
        retrieval_method: str = "bm25", 
        retrieval_topk: int = 10,
        index_path: str = "./index/bm25",
        corpus_path: str = "./data/corpus.jsonl",
        dataset_path: str = "./data",
        data_split: str = "train",
        faiss_gpu: bool = True,
        retrieval_model_path: str = "./model",
        retrieval_pooling_method: str = "mean",
        retrieval_query_max_length: int = 256,
        retrieval_use_fp16: bool = False,
        retrieval_batch_size: int = 256,  # Increase the default batch size
        max_content_tokens: int = 500
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.faiss_gpu = faiss_gpu
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size
        self.max_content_tokens = max_content_tokens


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI()

@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """
    Endpoint that accepts queries and performs retrieval.
    Input format:
    {
      "queries": ["What is Python?", "Tell me about neural networks."],
      "topk": 3,
      "return_scores": true
    }
    """
    if not request.topk:
        request.topk = config.retrieval_topk  # fallback to default

    # Perform batch retrieval
    results, scores = retriever.batch_search(
        query_list=request.queries,
        num=request.topk,
        return_score=request.return_scores
    )
    
    # Format response
    resp = []
    for i, single_result in enumerate(results):
        if request.return_scores:
            # If scores are returned, combine them with results
            combined = []
            for doc, score in zip(single_result, scores[i]):
                combined.append({"document": doc, "score": score})
            resp.append(combined)
        else:
            resp.append(single_result)
    return {"result": resp}

@app.get("/health")
def health_endpoint():
    return {"status": "ok"}

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Launch the local faiss retriever.")
    parser.add_argument("--index_path", type=str, default="/home/peterjin/mnt/index/wiki-18/e5_Flat.index", help="Corpus indexing file.")
    parser.add_argument("--corpus_path", type=str, default="/home/peterjin/mnt/data/retrieval-corpus/wiki-18.jsonl", help="Local corpus file.")
    parser.add_argument("--topk", type=int, default=3, help="Number of retrieved passages for one query.")
    parser.add_argument("--retriever_name", type=str, default="e5", help="Name of the retriever model.")
    parser.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2", help="Path of the retriever model.")
    parser.add_argument('--faiss_gpu', action='store_true', help='Use GPU for computation')
    parser.add_argument("--max_content_tokens", type=int, default=500, help="Maximum tokens per document content.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for dense retrieval.")
    parser.add_argument("--gpu_devices", type=str, default=None, help="Comma-separated GPU device IDs to use (e.g., '0,1,2,3').")

    args = parser.parse_args()
    
    # Configure the GPU devices if requested
    if args.gpu_devices:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
        print(f"Using GPU devices: {args.gpu_devices}")
    
    # 1) Build a config (could also parse from arguments).
    #    In real usage, you'd parse your CLI arguments or environment variables.
    config = Config(
        retrieval_method = args.retriever_name,  # or "dense"
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.topk,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=args.batch_size,
        max_content_tokens=args.max_content_tokens,
    )

    # 2) Instantiate a global retriever so it is loaded once and reused.
    retriever = get_retriever(config)
    
    # 3) Launch the server. By default, it listens on http://127.0.0.1:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
