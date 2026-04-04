"""
End-to-End Demo — Document Ingestion + Knowledge Pyramid + Retrieval

This script demonstrates the complete Part 1 pipeline:
    1. Load a sample document
    2. Chunk it using the sliding window strategy
    3. Build the 4-layer Knowledge Pyramid
    4. Query the pyramid and display ranked results

Run:
    python examples/demo_pipeline.py
"""

import os
import sys
from pathlib import Path

from part1_document_pipeline.input_layer import load_document
from part1_document_pipeline.chunker import chunk_document
from part1_document_pipeline.pyramid_builder import build_pyramid
from part1_document_pipeline.retriever import query_pyramid, format_results
from shared.config_manager import load_config


def main():
    """Run the complete document ingestion + retrieval demo."""
    
    print("\n" + "═" * 70)
    print("   VEXOO LABS — Document Ingestion + Knowledge Pyramid Demo")
    print("═" * 70)
    
    # ── Load Configuration ──
    config = load_config()
    
    # ── Step 1: Load Document ──
    print("\n[Step 1] Loading document...")
    sample_path = Path(__file__).parent / "sample_document.txt"
    
    if os.path.exists(sample_path):
        document = load_document(sample_path)
    else:
        # Fallback: use inline text
        print("  (Sample file not found, using inline text)")
        document = load_document(
            "Nexagen Technologies reported revenue of $847 million. "
            "The engineering team deployed 47 product updates. "
            "The legal department resolved 14 patent disputes."
        )
    
    print(f"  Document ID: {document.doc_id}")
    print(f"  Source: {document.source}")
    print(f"  Characters: {document.char_count}")
    print(f"  Est. Tokens: {document.token_count}")
    
    # ── Step 2: Chunk Document ──
    print(f"\n[Step 2] Chunking with sliding window...")
    print(f"  Window size: {config.chunker.window_size_chars} chars")
    print(f"  Overlap: {config.chunker.overlap_ratio * 100:.0f}%")
    
    chunks = chunk_document(document, config.chunker)
    
    print(f"  Chunks created: {len(chunks)}")
    for chunk in chunks:
        print(f"    {chunk.chunk_id}: {chunk.start_char}-{chunk.end_char} "
              f"({len(chunk.text)} chars, ~{chunk.token_count} tokens)")
    
    # ── Step 3: Build Knowledge Pyramid ──
    print(f"\n[Step 3] Building Knowledge Pyramid (4 layers)...")
    pyramid = build_pyramid(chunks, config.pyramid)
    
    print(f"  Pyramid nodes: {len(pyramid)}")
    print(f"  Total entries: {len(pyramid) * 4} (nodes × 4 layers)")
    
    # Display pyramid summary for each chunk
    for chunk_id, node in pyramid.items():
        print(f"\n  ── {chunk_id} ──")
        print(f"  L1 (Raw):      {len(node.raw_text)} chars")
        print(f"  L2 (Summary):  {node.summary[:80]}...")
        print(f"  L3 (Category): {node.category} "
              f"(confidence: {node.category_confidence:.3f})")
        print(f"  L4 (Keywords): {', '.join(node.keywords[:5])}")
    
    # ── Step 4: Query the Pyramid ──
    print(f"\n[Step 4] Running queries against the pyramid...")
    
    # Sample queries targeting different domains and pyramid levels
    test_queries = [
        "What are the key financial metrics and revenue figures?",
        "Describe the technical architecture and system performance",
        "What legal compliance measures are in place?",
        "How is the healthcare AI system performing?",
        "What is the company's market strategy and growth plan?",
    ]
    
    for query in test_queries:
        results = query_pyramid(query, pyramid, config.retriever)
        print(format_results(results, query))
    
    # ── Interactive Mode ──
    print("\n" + "─" * 70)
    print("  Interactive Query Mode (type 'quit' to exit)")
    print("─" * 70)
    
    while True:
        try:
            user_query = input("\n  Your query: ").strip()
            if user_query.lower() in ("quit", "exit", "q"):
                print("  Goodbye!")
                break
            if not user_query:
                continue
            
            results = query_pyramid(user_query, pyramid, config.retriever)
            print(format_results(results, user_query))
        except (KeyboardInterrupt, EOFError):
            print("\n  Goodbye!")
            break


if __name__ == "__main__":
    main()
