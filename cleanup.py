#!/usr/bin/env python3
"""
Cleanup script for RAG Any File system.
Deletes all saved embeddings, documents, and uploaded files data.
"""

import os
import shutil
import argparse
import sys

def cleanup_data_directory(data_dir='data', confirm=True):
    """Clean up the data directory containing embeddings and metadata."""
    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' does not exist.")
        return

    files_to_delete = [
        'faiss_index.idx',
        'documents.json',
        'uploaded_files.json'
    ]

    deleted_files = []
    for filename in files_to_delete:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            if confirm:
                response = input(f"Delete {filepath}? (y/N): ")
                if response.lower() not in ['y', 'yes']:
                    continue
            os.remove(filepath)
            deleted_files.append(filepath)
            print(f"Deleted: {filepath}")
        else:
            print(f"File not found: {filepath}")

    return deleted_files

def cleanup_uploads_directory(uploads_dir='uploads', confirm=True):
    """Clean up the uploads directory containing uploaded PDF files."""
    if not os.path.exists(uploads_dir):
        print(f"Uploads directory '{uploads_dir}' does not exist.")
        return

    if confirm:
        response = input(f"Delete ALL files in {uploads_dir}? This cannot be undone! (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Upload cleanup cancelled.")
            return

    deleted_files = []
    for filename in os.listdir(uploads_dir):
        filepath = os.path.join(uploads_dir, filename)
        if os.path.isfile(filepath):
            os.remove(filepath)
            deleted_files.append(filepath)
            print(f"Deleted: {filepath}")

    return deleted_files

def main():
    parser = argparse.ArgumentParser(description='Clean up RAG Any File data')
    parser.add_argument('--data-dir', default='data', help='Data directory path (default: data)')
    parser.add_argument('--uploads-dir', default='uploads', help='Uploads directory path (default: uploads)')
    parser.add_argument('--yes', action='store_true', help='Skip confirmation prompts')
    parser.add_argument('--embeddings-only', action='store_true', help='Only delete embeddings and metadata, keep uploaded files')
    parser.add_argument('--uploads-only', action='store_true', help='Only delete uploaded files, keep embeddings and metadata')

    args = parser.parse_args()

    confirm = not args.yes

    print("RAG Any File Cleanup Script")
    print("=" * 30)

    total_deleted = []

    # Clean up data directory (embeddings and metadata)
    if not args.uploads_only:
        print("\nCleaning up data directory...")
        deleted_data = cleanup_data_directory(args.data_dir, confirm)
        if deleted_data:
            total_deleted.extend(deleted_data)

    # Clean up uploads directory
    if not args.embeddings_only:
        print("\nCleaning up uploads directory...")
        deleted_uploads = cleanup_uploads_directory(args.uploads_dir, confirm)
        if deleted_uploads:
            total_deleted.extend(deleted_uploads)

    print(f"\nCleanup complete! Deleted {len(total_deleted)} files.")
    if total_deleted:
        print("Deleted files:")
        for filepath in total_deleted:
            print(f"  - {filepath}")

if __name__ == '__main__':
    main()