import pandas as pd
import json
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Optional
from tqdm import tqdm
import sys
import logging

current_dir = Path(__file__).parent
root_dir = current_dir.parent
sys.path.insert(0, str(root_dir))  # Add parent directory (root)
sys.path.insert(0, str(current_dir))  # Add current directory

from pathology_agent_streaming_image_tool_manual_context import PathologyAgentStreaming


ISO = "%Y-%m-%dT%H:%M:%S.%fZ"

REVIEW_PROMPT = """You are being tasked with comprehensively analyzing the following whole-slide-image. You must analyze the tissue and determine if it is normal or cancerous tissue. You need to explore the slide and document which regions are important for your analysis. Finally, give your best prediction for the organ type and diagnosis.

CRITICAL: You have a limited number of iterations. You MUST submit your final response BEFORE reaching the maximum iteration limit. Reserve your final iteration for submitting the JSON response below. DO NOT exceed the iteration limit or your analysis will be incomplete.

When you have completed your analysis, provide documentation of the 5 most important regions you explored that helped you reach your answer.

Respond in the following JSON format:
```json
{{
    "regions": [
        {{
            "region_number": 1,
            "bbox": {{
                "x": <x_coordinate>,
                "y": <y_coordinate>,
                "width": <width>,
                "height": <height>
            }},
            "why_zoomed": "<explanation of why you zoomed to this region>",
            "observations": "<what you observed in this region>",
            "diagnostic_relevance": "<how this relates to your diagnosis>"
        }},
        {{
            "region_number": 2,
            "bbox": {{
                "x": <x_coordinate>,
                "y": <y_coordinate>,
                "width": <width>,
                "height": <height>
            }},
            "why_zoomed": "<explanation of why you zoomed to this region>",
            "observations": "<what you observed in this region>",
            "diagnostic_relevance": "<how this relates to your diagnosis>"
        }}
        // ... continue for all 5 regions
    ],
    "answer": {{
        "organ_type": "<predicted organ type>",
        "diagnosis": "<normal or specific cancer type>",
        "confidence": "<high/medium/low>",
        "key_findings": "<brief summary of key pathological findings that support your diagnosis>"
    }}
}}
```

REMINDER: Track your iteration count carefully and submit this final JSON response before exceeding the maximum number of iterations allowed.
"""


@dataclass
class AgentResult:
    # Inputs/keys
    file_id: Optional[str]
    file_name: str
    conv_id: str
    prompt: str
    benchmark: str

    # Outputs
    status: str                           # "ok" | "missing" | "error" | "timeout" | "max_retries_exceeded"
    prediction: Optional[str] = None      # LLM output/prediction
    error: Optional[str] = None
    raw_responses: Optional[list] = None  # Raw response dictionaries from each iteration

    # Metadata
    model_name: Optional[str] = None
    start_time: str = ""
    end_time: str = ""
    duration_s: float = 0.0
    retry_count: int = 0                  # Number of retries attempted
    script_version: str = "path_review_v1"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime(ISO)


def stream_final(agent: PathologyAgentStreaming, prompt: str, timeout_s: Optional[float]) -> dict:
    """
    Consume streaming updates until 'final' or 'error'.
    Returns {"status": "ok", "prediction": "...", "raw_responses": [...], "max_iterations_reached": bool} 
    or {"status": "error"/"timeout", "error": "..."}.
    """
    start = time.time()
    try:
        it = agent.analyze_streaming(prompt)
        while True:
            if timeout_s and (time.time() - start) > timeout_s:
                return {"status": "timeout", "error": f"Timed out after {timeout_s}s", "max_iterations_reached": False}
            try:
                update = next(it)
            except StopIteration:
                return {"status": "error", "error": "Stream ended without final result", "max_iterations_reached": False}
            except Exception as e:
                return {"status": "error", "error": f"Stream error: {e!r}", "max_iterations_reached": False}

            utype = update.get("type")
            if utype == "final":
                return {
                    "status": "ok", 
                    "prediction": update.get("analysis"),
                    "raw_responses": update.get("raw_responses", []),
                    "max_iterations_reached": update.get("max_iterations_reached", False)
                }
            if utype == "error":
                return {"status": "error", "error": update.get("message") or "Unknown agent error", "max_iterations_reached": False}
    except Exception as e:
        return {"status": "error", "error": f"Agent exception: {e!r}", "max_iterations_reached": False}


def process_one_image(row, filedir, dataset_name, max_iterations, model, max_images_in_context, timeout_s, max_retries=3):
    """Process a single image and return AgentResult. Retries up to max_retries times if max iterations exceeded."""
    file_name = row["file_name"]
    file_id = row.get("Tissue Sample ID", row.get("image", file_name))
    wsi_path = filedir / file_name
    
    start_ts = now_iso()
    t0 = time.time()
    base_conv_id = str(uuid.uuid4())
    
    # Check if file exists
    if not wsi_path.exists():
        result = AgentResult(
            file_id=str(file_id),
            file_name=file_name,
            conv_id=base_conv_id,
            prompt=REVIEW_PROMPT,
            benchmark=dataset_name,
            status="missing",
            start_time=start_ts,
            end_time=now_iso(),
            duration_s=time.time() - t0,
            model_name=model,
            retry_count=0
        )
        return result
    
    # Retry loop
    retry_count = 0
    last_error = None
    all_raw_responses = []
    
    for attempt in range(max_retries + 1):  # +1 for initial attempt
        conv_id = f"{base_conv_id}_attempt{attempt}" if attempt > 0 else base_conv_id
        
        # Initialize agent
        agent = PathologyAgentStreaming(
            wsi_file=str(wsi_path),
            session_id=conv_id,
            max_iterations=max_iterations,
            model=model,
            max_images_in_context=max_images_in_context
        )
        
        # Run agent
        out = stream_final(agent, REVIEW_PROMPT, timeout_s=timeout_s)
        
        status = out["status"]
        prediction = out.get("prediction")
        error = out.get("error")
        raw_responses = out.get("raw_responses", [])
        max_iterations_reached = out.get("max_iterations_reached", False)
        
        # Store responses from this attempt
        all_raw_responses.extend(raw_responses)
        
        # If successful or non-retryable error, return immediately
        if status == "ok" and not max_iterations_reached:
            result = AgentResult(
                file_id=str(file_id),
                file_name=file_name,
                conv_id=base_conv_id,
                prompt=REVIEW_PROMPT,
                benchmark=dataset_name,
                status=status,
                prediction=prediction,
                error=error,
                raw_responses=all_raw_responses,
                model_name=model,
                start_time=start_ts,
                end_time=now_iso(),
                duration_s=time.time() - t0,
                retry_count=retry_count
            )
            return result
        
        # If timeout or other error (not max iterations), don't retry
        if status in ["timeout", "error"] and not max_iterations_reached:
            result = AgentResult(
                file_id=str(file_id),
                file_name=file_name,
                conv_id=base_conv_id,
                prompt=REVIEW_PROMPT,
                benchmark=dataset_name,
                status=status,
                prediction=prediction,
                error=error,
                raw_responses=all_raw_responses,
                model_name=model,
                start_time=start_ts,
                end_time=now_iso(),
                duration_s=time.time() - t0,
                retry_count=retry_count
            )
            return result
        
        # Max iterations reached - retry if we have attempts left
        if max_iterations_reached and attempt < max_retries:
            retry_count += 1
            last_error = f"Max iterations reached on attempt {attempt + 1}"
            logging.warning(f"{file_name}: Max iterations reached, retrying ({retry_count}/{max_retries})...")
            continue
        
        # If we get here and max iterations reached, we've exhausted retries
        if max_iterations_reached:
            result = AgentResult(
                file_id=str(file_id),
                file_name=file_name,
                conv_id=base_conv_id,
                prompt=REVIEW_PROMPT,
                benchmark=dataset_name,
                status="max_retries_exceeded",
                prediction=prediction,
                error=f"Max iterations exceeded after {retry_count} retries",
                raw_responses=all_raw_responses,
                model_name=model,
                start_time=start_ts,
                end_time=now_iso(),
                duration_s=time.time() - t0,
                retry_count=retry_count
            )
            return result
    
    # Fallback (should not reach here)
    result = AgentResult(
        file_id=str(file_id),
        file_name=file_name,
        conv_id=base_conv_id,
        prompt=REVIEW_PROMPT,
        benchmark=dataset_name,
        status="error",
        prediction=None,
        error="Unexpected error in retry loop",
        raw_responses=all_raw_responses,
        model_name=model,
        start_time=start_ts,
        end_time=now_iso(),
        duration_s=time.time() - t0,
        retry_count=retry_count
    )
    return result


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    
    # Dataset configurations
    datasets = [
        {
            "name": "gtex",
            "dataset": "/home/thb286/PathAgent/gtex/gtex_organ_type_test.csv",
            "filedir": "/n/data1/hms/dbmi/manrai/PathAgent/gtex_organ_type_test"
        },
        {
            "name": "tcga",
            "dataset": "/home/thb286/PathAgent/tcga_ut_benchmark/csv_datasets/test_llm_filtered.csv",
            "filedir": "/n/data1/hms/dbmi/manrai/PathAgent/tcga_data"
        }
    ]
    
    # Configuration
    SEED = 42
    NUM_SAMPLES = 25
    MAX_ITERATIONS = 20
    MODEL = "gpt-5"
    MAX_IMAGES_IN_CONTEXT = None
    TIMEOUT_S = 900.0
    MAX_WORKERS = 30  # Number of parallel workers
    MAX_RETRIES = 3  # Number of times to retry if max iterations exceeded (set to 0 to disable retries)
    
    # Corrupt/missing slides to exclude from GTEX dataset
    GTEX_EXCLUDE_FILES = [
        "GTEX-13QJ3-2726.tiff",
        "GTEX-1LGOU-0226.tiff",
        "GTEX-1NV88-0326.tiff",
        "GTEX-1269W-2626.tiff"
    ]
    
    # Output directory
    output_dir = Path("/home/thb286/PathAgent/results/path_review3")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each dataset
    for dataset_config in datasets:
        dataset_name = dataset_config["name"]
        dataset_path = dataset_config["dataset"]
        filedir = Path(dataset_config["filedir"])
        
        print(f"\n{'='*80}")
        print(f"Processing {dataset_name.upper()} dataset")
        print(f"{'='*80}\n")
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        
        # Extract file_name column (handle different column names)
        if "image_path" in df.columns:
            df["file_name"] = df["image_path"].apply(lambda x: Path(x).name)
        elif "file_name" not in df.columns:
            print(f"Warning: Could not find image path column in {dataset_name}")
            continue
        
        # Exclude corrupt/missing slides for GTEX dataset
        if dataset_name == "gtex":
            initial_count = len(df)
            df = df[~df["file_name"].isin(GTEX_EXCLUDE_FILES)].reset_index(drop=True)
            excluded_count = initial_count - len(df)
            if excluded_count > 0:
                print(f"Excluded {excluded_count} corrupt/missing slides from {dataset_name}")
        
        # Sample 25 random images with seed
        if len(df) > NUM_SAMPLES:
            df_sampled = df.sample(n=NUM_SAMPLES, random_state=SEED).reset_index(drop=True)
        else:
            df_sampled = df.copy()
            print(f"Warning: Dataset has only {len(df)} samples, using all of them")
        
        print(f"Sampled {len(df_sampled)} images from {dataset_name}")
        
        # Output file
        output_file = output_dir / f"{dataset_name}_path_review_results.jsonl"
        
        # Convert to list of dicts
        rows = [row.to_dict() for _, row in df_sampled.iterrows()]
        
        def _write_jsonl(res):
            with output_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(res), ensure_ascii=False) + "\n")
        
        # Process images
        if MAX_WORKERS == 1:
            # Sequential processing
            for row in tqdm(rows, desc=f"Processing {dataset_name}"):
                result = process_one_image(row, filedir, dataset_name, MAX_ITERATIONS, MODEL, MAX_IMAGES_IN_CONTEXT, TIMEOUT_S, MAX_RETRIES)
                _write_jsonl(result)
                retry_info = f" (retries: {result.retry_count})" if result.retry_count > 0 else ""
                tqdm.write(f"{result.file_name} -> {result.status} ({result.duration_s:.2f}s){retry_info}")
        else:
            # Parallel processing
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(process_one_image, row, filedir, dataset_name, MAX_ITERATIONS, MODEL, MAX_IMAGES_IN_CONTEXT, TIMEOUT_S, MAX_RETRIES): row
                    for row in rows
                }
                
                with tqdm(total=len(rows), desc=f"Processing {dataset_name}") as pbar:
                    for future in as_completed(futures):
                        result = future.result()
                        _write_jsonl(result)
                        retry_info = f" (retries: {result.retry_count})" if result.retry_count > 0 else ""
                        tqdm.write(f"{result.file_name} -> {result.status} ({result.duration_s:.2f}s){retry_info}")
                        pbar.update(1)
        
        print(f"\n{dataset_name} results saved to: {output_file}")
    
    print(f"\n{'='*80}")
    print(f"ALL PROCESSING COMPLETE")
    print(f"{'='*80}\n")
