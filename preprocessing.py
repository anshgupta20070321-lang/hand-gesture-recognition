import numpy as np
import cv2
import os
from image_processing import preprocess_image
from tqdm import tqdm

IMG_SIZE = 128

class DataPreprocessor:
    def __init__(self, source_dir='data', target_dir='data2'):
        self.source_dir = source_dir
        self.target_dir = target_dir
        
    def setup_directories(self):
        """Create directory structure for preprocessed data"""
        for split in ['train', 'test']:
            split_path = os.path.join(self.target_dir, split)
            if not os.path.exists(split_path):
                os.makedirs(split_path)
            
            # Create subdirectories for each class
            source_split_path = os.path.join(self.source_dir, split)
            if os.path.exists(source_split_path):
                for class_dir in os.listdir(source_split_path):
                    class_path = os.path.join(split_path, class_dir)
                    if not os.path.exists(class_path):
                        os.makedirs(class_path)
    
    def process_dataset(self, train_test_split=0.0):
        """
        Process all images in the dataset
        
        Args:
            train_test_split: If > 0, split data into train/test (e.g., 0.2 for 20% test)
                             If 0, process existing train/test folders separately
        """
        self.setup_directories()
        
        stats = {
            'train': {},
            'test': {}
        }
        
        # Process each split
        for split in ['train', 'test']:
            source_split_path = os.path.join(self.source_dir, split)
            target_split_path = os.path.join(self.target_dir, split)
            
            if not os.path.exists(source_split_path):
                print(f"Warning: {source_split_path} does not exist, skipping...")
                continue
            
            print(f"\nProcessing {split} data...")
            
            # Get all class directories
            class_dirs = sorted([d for d in os.listdir(source_split_path) 
                               if os.path.isdir(os.path.join(source_split_path, d))])
            
            for class_dir in class_dirs:
                source_class_path = os.path.join(source_split_path, class_dir)
                target_class_path = os.path.join(target_split_path, class_dir)
                
                # Get all images
                images = [f for f in os.listdir(source_class_path) 
                         if f.endswith(('.jpg', '.jpeg', '.png'))]
                
                print(f"  Processing class '{class_dir}': {len(images)} images")
                
                processed_count = 0
                failed_count = 0
                
                # Process each image with progress bar
                for img_file in tqdm(images, desc=f"  {class_dir}", leave=False):
                    try:
                        source_img_path = os.path.join(source_class_path, img_file)
                        target_img_path = os.path.join(target_class_path, img_file)
                        
                        # Preprocess image
                        processed_img = preprocess_image(source_img_path, 
                                                        target_size=(IMG_SIZE, IMG_SIZE))
                        
                        # Save preprocessed image
                        cv2.imwrite(target_img_path, processed_img)
                        processed_count += 1
                        
                    except Exception as e:
                        print(f"    Error processing {img_file}: {str(e)}")
                        failed_count += 1
                
                stats[split][class_dir] = {
                    'processed': processed_count,
                    'failed': failed_count
                }
        
        return stats
    
    def print_statistics(self, stats):
        """Print processing statistics"""
        print("\n" + "="*60)
        print("PREPROCESSING SUMMARY")
        print("="*60)
        
        for split in ['train', 'test']:
            if stats[split]:
                print(f"\n{split.upper()} DATA:")
                print("-" * 60)
                
                total_processed = 0
                total_failed = 0
                
                for class_name, class_stats in sorted(stats[split].items()):
                    processed = class_stats['processed']
                    failed = class_stats['failed']
                    total_processed += processed
                    total_failed += failed
                    
                    print(f"  Class {class_name:10s}: "
                          f"{processed:4d} processed, "
                          f"{failed:2d} failed")
                
                print("-" * 60)
                print(f"  Total:           {total_processed:4d} processed, "
                      f"{total_failed:2d} failed")
        
        print("="*60)
    
    def verify_data_balance(self):
        """Check class balance in the dataset"""
        print("\n" + "="*60)
        print("DATA BALANCE CHECK")
        print("="*60)
        
        for split in ['train', 'test']:
            split_path = os.path.join(self.target_dir, split)
            
            if not os.path.exists(split_path):
                continue
            
            print(f"\n{split.upper()} DATA:")
            print("-" * 60)
            
            class_counts = {}
            for class_dir in sorted(os.listdir(split_path)):
                class_path = os.path.join(split_path, class_dir)
                if os.path.isdir(class_path):
                    count = len([f for f in os.listdir(class_path) 
                               if f.endswith(('.jpg', '.jpeg', '.png'))])
                    class_counts[class_dir] = count
            
            if class_counts:
                max_count = max(class_counts.values())
                min_count = min(class_counts.values())
                avg_count = sum(class_counts.values()) / len(class_counts)
                
                for class_name, count in sorted(class_counts.items()):
                    bar_length = int((count / max_count) * 40)
                    bar = "█" * bar_length
                    print(f"  Class {class_name:10s}: {count:4d} {bar}")
                
                print("-" * 60)
                print(f"  Min: {min_count:4d} | Max: {max_count:4d} | "
                      f"Avg: {avg_count:6.1f}")
                
                # Warn if imbalanced
                if max_count > 2 * min_count:
                    print("\n  ⚠️  WARNING: Dataset is imbalanced! "
                          "Consider collecting more data for underrepresented classes.")
        
        print("="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess hand gesture images')
    parser.add_argument('--source', default='data', 
                       help='Source directory (default: data)')
    parser.add_argument('--target', default='data2',
                       help='Target directory (default: data2)')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify data balance without processing')
    
    args = parser.parse_args()
    
    preprocessor = DataPreprocessor(args.source, args.target)
    
    if args.verify_only:
        preprocessor.verify_data_balance()
    else:
        print("Starting data preprocessing...")
        stats = preprocessor.process_dataset()
        preprocessor.print_statistics(stats)
        preprocessor.verify_data_balance()
        print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()
