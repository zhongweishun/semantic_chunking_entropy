"""Run the preserved Nov25 chunker on a UTF-8 text file using your model endpoint."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',default='meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8')
    parser.add_argument('--base-url',default=os.environ.get('OPENAI_BASE_URL'))
    parser.add_argument('--k',type=int,default=4)
    parser.add_argument('--enable-fallback',action='store_true')
    parser.add_argument('--multithread',action='store_true')
    args=parser.parse_args()
    if not args.base_url:parser.error('Set OPENAI_BASE_URL or provide --base-url')
    if args.k<2:parser.error('k must be at least 2')
    if args.output.exists():parser.error('Output already exists; choose a new output path')
    from openai import OpenAI
    from src.hierarchical_chunker_utilities_LlamaDS_25Nov2025 import main as chunk
    client=OpenAI(base_url=args.base_url,api_key=os.environ.get('OPENAI_API_KEY','unused'))
    result=chunk(args.input.read_text(encoding='utf-8'),model=args.model,client=client,
                 k=args.k,multithread=args.multithread,enable_fallback=args.enable_fallback,
                 debug=False,showfig=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(args.output)


if __name__=='__main__':main()
