#!/usr/bin/env python3
#
# Multi_Agent_RLM_0.2.py
# Proof-of-Concept script to investigate using RLM technique for multi-agent systems.
# PDF Document ingestioon and conversion to Markdown using the docling library.
# RLM Agent interaction with the converted Markdown. 

import re
from typing import List, Dict, Any
from openai import OpenAI
from docling.document_converter import DocumentConverter

# 1. Setup Clients
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

MODELS = {
    "scout": "nvidia/nemotron-3-nano",
    "architect": "qwen/qwen3-30b-a3b-thinking-2507",
    "worker": "openai/gpt-oss-20b"
}

# ----------------------------------------------------------------------
# 2. PDF Ingestion (Docling)
# https://pypi.org/project/docling/
def ingest_multiple_pdfs(filepaths: List[str]) -> str:
    """Converts a list of PDFs to a single structured Markdown context."""
    converter = DocumentConverter()
    combined_md = ""
    
    for path in filepaths:
        print(f"Ingesting: {path}...")
        result = converter.convert(path)
        md = result.document.export_to_markdown()
        combined_md += f"\n\n--- SOURCE: {path} ---\n\n" + md
        
    return combined_md

# ----------------------------------------------------------------------
# 3. Enhanced Dynamic REPL
class MultiAgentREPL:
    def __init__(self, context: str):
        self.context = context
        self.vars: Dict[str, Any] = {"context": context}

    def execute_command(self, cmd: str) -> str:
        # 1. Handle Assignments: var = llm_query(...)
        if " = llm_query(" in cmd:
            var_name, query_raw = cmd.split(" = llm_query(")
            query_template = query_raw.rstrip(")")
            
            # Resolve placeholders like {context[0:1000]} or {prev_step}
            prompt = self.resolve_placeholders(query_template)
            
            # Use the 'Worker' model for standard chunk processing
            response = client.chat.completions.create(
                model=MODELS["worker"],
                messages=[{"role": "user", "content": prompt}]
            )
            result = response.choices[0].message.content
            self.vars[var_name.strip()] = result
            return f"Updated {var_name.strip()}"

        # 2. Handle Final Extraction: FINAL(var_name)
        if cmd.startswith("FINAL("):
            var_name = cmd[6:-1].strip()
            return self.vars.get(var_name, f"Error: {var_name} not found.")

        return "Command executed."

    def resolve_placeholders(self, template: str) -> str:
        resolved = template
        # Handle context slicing: {context[start:end]}
        slices = re.findall(r"\{context\[(\d+):(\d+)\]\}", resolved)
        for start, end in slices:
            chunk = self.context[int(start):int(end)]
            resolved = resolved.replace(f"{{context[{start}:{end}]}}", chunk)
        
        # Handle variable injection: {var_name}
        for v_name, v_val in self.vars.items():
            if f"{{{v_name}}}" in resolved:
                resolved = resolved.replace(f"{{{v_name}}}", str(v_val))
        return resolved

# ----------------------------------------------------------------------
# 4. The Autonomous Controller
class RLMController:
    def __init__(self, pdf_paths: List[str]):
        self.context = ingest_multiple_pdfs(pdf_paths)
        self.repl = MultiAgentREPL(self.context)

    def solve(self, goal: str):
        # Phase 1: Scouting (Nemotron-3-Nano)
        print("\n[PHASE 1] SCOUTING CONTEXT DENSITY...")
        scout_prompt = f"Analyze this context ({len(self.context)} chars) and suggest a chunk size (500-5000) for this goal: {goal}"
        strategy = client.chat.completions.create(model=MODELS["scout"], messages=[{"role": "user", "content": scout_prompt}]).choices[0].message.content

        # Phase 2: Planning (Qwen3-Thinking)
        print("[PHASE 2] GENERATING LOGICAL PLAN...")
        plan_prompt = f"Goal: {goal}\nStrategy: {strategy}\nContext Len: {len(self.context)}\nOutput a multi-step plan using 'var = llm_query(...)' and 'FINAL(var)'. Provide reasoning in <think> tags."
        plan_res = client.chat.completions.create(
            model=MODELS["architect"], 
            messages=[{"role": "system", "content": "You are a logical planner."}, {"role": "user", "content": plan_prompt}]
        ).choices[0].message.content
        
        # Phase 3: Execution
        print("[PHASE 3] EXECUTING PLAN...")
        plan_lines = [l.strip() for l in plan_res.split('\n') if "=" in l or "FINAL(" in l]
        
        final_output = ""
        for line in plan_lines:
            res = self.repl.execute_command(line)
            if "FINAL(" in line:
                final_output = res # This is our Markdown report
        
        return final_output

# ----------------------------------------------------------------------
# 5. Execution
if __name__ == "__main__":
    # Point to your local PDF files
    files = ["report_a.pdf", "report_b.pdf"] 
    
    agent = RLMController(files)
    markdown_report = agent.solve("Synthesize a comparison of the technical risks in both documents.")
    
    print("\n" + "="*50)
    print("FINAL CONSOLIDATED REPORT (MARKDOWN)")
    print("="*50 + "\n")
    print(markdown_report)
