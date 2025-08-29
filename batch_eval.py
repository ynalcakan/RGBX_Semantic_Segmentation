import os
import cv2
import argparse
import numpy as np
from PIL import Image
from collections import defaultdict
import json
import time

# Optional dependencies for advanced reporting and visualization
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    sns = None
    MATPLOTLIB_AVAILABLE = False

import torch
import torch.nn as nn

from config import config
from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualize import print_iou, show_img
from engine.evaluator import Evaluator
from engine.logger import get_logger
from utils.metric import hist_info, compute_score
from dataloader.RGBXDataset import RGBXDataset
from models.builder import EncoderDecoder as segmodel
from dataloader.dataloader import ValPre

logger = get_logger()

class BatchSegEvaluator(Evaluator):
    """Extended evaluator for batch evaluation of multiple checkpoints"""
    
    def func_per_iteration(self, data, device):
        img = data['data']
        label = data['label']
        modal_x = data['modal_x']
        name = data['fn']
        pred = self.sliding_eval_rgbX(img, modal_x, config.eval_crop_size, config.eval_stride_rate, device)
        hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes, pred, label)
        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp, 'correct': correct_tmp}

        if self.save_path is not None:
            ensure_dir(self.save_path)
            ensure_dir(self.save_path+'_color')

            fn = name + '.png'

            # save colored result
            result_img = Image.fromarray(pred.astype(np.uint8), mode='P')
            class_colors = self.dataset.get_class_colors()
            palette_list = list(np.array(class_colors).flat)
            if len(palette_list) < 768:
                palette_list += [0] * (768 - len(palette_list))
            result_img.putpalette(palette_list)
            result_img.save(os.path.join(self.save_path+'_color', fn))

            # save raw result
            cv2.imwrite(os.path.join(self.save_path, fn), pred)

        return results_dict

    def compute_metric(self, results, return_raw_metrics=False):
        """Compute metrics and return both formatted string and raw values"""
        hist = np.zeros((config.num_classes, config.num_classes))
        correct = 0
        labeled = 0
        count = 0
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1

        # Calculate metrics from histogram
        iou, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(hist, correct, labeled)
        
        if return_raw_metrics:
            return {
                'iou': iou,
                'mean_IoU': mean_IoU,
                'freq_IoU': freq_IoU,
                'mean_pixel_acc': mean_pixel_acc,
                'pixel_acc': pixel_acc,
                'hist': hist,
                'correct': correct,
                'labeled': labeled
            }
        
        # Get the formatted result string (without printing)
        result_line = print_iou(iou, freq_IoU, mean_pixel_acc, pixel_acc,
                              config.class_names, show_no_back=False, no_print=True)
        
        return result_line

class BatchEvaluationRunner:
    """Main class for running batch evaluation and generating comprehensive reports"""
    
    def __init__(self, devices='0', verbose=False, save_results_dir=None):
        self.devices = parse_devices(devices)
        self.verbose = verbose
        self.save_results_dir = save_results_dir or 'batch_evaluation_results'
        ensure_dir(self.save_results_dir)
        
        # Initialize network and dataset
        self.network = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
        self.setup_dataset()
        
        # Results storage
        self.all_results = {}
        self.checkpoint_epochs = list(range(config.checkpoint_start_epoch, config.nepochs + 1, config.checkpoint_step))
        
    def setup_dataset(self):
        """Setup dataset for evaluation"""
        data_setting = {
            'rgb_root': config.rgb_root_folder,
            'rgb_format': config.rgb_format,
            'gt_root': config.gt_root_folder,
            'gt_format': config.gt_format,
            'transform_gt': config.gt_transform,
            'x_root': config.x_root_folder,
            'x_format': config.x_format,
            'x_single_channel': config.x_is_single_channel,
            'class_names': config.class_names,
            'train_source': config.train_source,
            'eval_source': config.eval_source,
            'class_counts': config.num_classes
        }
        val_pre = ValPre()
        self.dataset = RGBXDataset(data_setting, 'val', val_pre)
        
    def evaluate_checkpoint(self, epoch):
        """Evaluate a single checkpoint"""
        checkpoint_path = os.path.join(config.checkpoint_dir, f'epoch-{epoch}.pth')
        
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return None
            
        logger.info(f"Evaluating checkpoint: epoch-{epoch}.pth")
        
        with torch.no_grad():
            evaluator = BatchSegEvaluator(
                self.dataset, config.num_classes, config.norm_mean,
                config.norm_std, self.network,
                config.eval_scale_array, config.eval_flip,
                self.devices, self.verbose, save_path=None, show_image=False
            )
            
            # Load model
            evaluator.val_func = load_model(evaluator.network, checkpoint_path)
            
            # Run evaluation
            start_time = time.time()
            if len(self.devices) == 1:
                all_results = []
                for idx in range(evaluator.ndata):
                    dd = evaluator.dataset[idx]
                    results_dict = evaluator.func_per_iteration(dd, self.devices[0])
                    all_results.append(results_dict)
                    
                    if self.verbose and (idx + 1) % 50 == 0:
                        logger.info(f"Processed {idx + 1}/{evaluator.ndata} images")
            else:
                # Multi-GPU evaluation would go here
                raise NotImplementedError("Multi-GPU batch evaluation not implemented yet")
            
            # Compute metrics
            raw_metrics = evaluator.compute_metric(all_results, return_raw_metrics=True)
            eval_time = time.time() - start_time
            
            logger.info(f"Epoch {epoch} evaluation completed in {eval_time:.2f}s")
            
            return {
                'epoch': epoch,
                'metrics': raw_metrics,
                'eval_time': eval_time,
                'checkpoint_path': checkpoint_path
            }
    
    def run_batch_evaluation(self):
        """Run evaluation on all checkpoints"""
        logger.info("Starting batch evaluation...")
        logger.info(f"Evaluating checkpoints: {self.checkpoint_epochs}")
        
        for epoch in self.checkpoint_epochs:
            result = self.evaluate_checkpoint(epoch)
            if result:
                self.all_results[epoch] = result
                logger.info(f"✓ Epoch {epoch}: mIoU = {result['metrics']['mean_IoU']:.4f}")
            else:
                logger.warning(f"✗ Failed to evaluate epoch {epoch}")
        
        if not self.all_results:
            logger.error("No successful evaluations. Check checkpoint directory.")
            return
            
        # Generate comprehensive report
        self.generate_comprehensive_report()
        
    def generate_comprehensive_report(self):
        """Generate comprehensive evaluation report"""
        logger.info("Generating comprehensive evaluation report...")
        
        # Create summary data structures
        summary_data = []
        per_class_data = []
        
        for epoch, result in sorted(self.all_results.items()):
            metrics = result['metrics']
            
            # Summary metrics
            summary_data.append({
                'Epoch': epoch,
                'mIoU': metrics['mean_IoU'] * 100,
                'Freq_IoU': metrics['freq_IoU'] * 100,
                'Mean_Pixel_Acc': metrics['mean_pixel_acc'] * 100,
                'Pixel_Acc': metrics['pixel_acc'] * 100,
                'Eval_Time': result['eval_time']
            })
            
            # Per-class IoU
            for class_idx, class_iou in enumerate(metrics['iou']):
                per_class_data.append({
                    'Epoch': epoch,
                    'Class_ID': class_idx,
                    'Class_Name': config.class_names[class_idx],
                    'IoU': class_iou * 100
                })
        
        # Create DataFrames if pandas is available
        if PANDAS_AVAILABLE:
            summary_df = pd.DataFrame(summary_data)
            per_class_df = pd.DataFrame(per_class_data)
            
            # Save raw data
            summary_df.to_csv(os.path.join(self.save_results_dir, 'summary_metrics.csv'), index=False)
            per_class_df.to_csv(os.path.join(self.save_results_dir, 'per_class_iou.csv'), index=False)
            
            # Generate text report
            self.generate_text_report(summary_df, per_class_df)
            
            # Generate visualizations
            if MATPLOTLIB_AVAILABLE:
                self.generate_visualizations(summary_df, per_class_df)
            else:
                logger.warning("Matplotlib not available. Skipping visualizations.")
                
        else:
            logger.warning("Pandas not available. Using basic reporting.")
            self.generate_basic_report(summary_data, per_class_data)
        
        # Save detailed results as JSON
        self.save_detailed_results()
        
    def generate_text_report(self, summary_df, per_class_df):
        """Generate detailed text report"""
        report_path = os.path.join(self.save_results_dir, 'evaluation_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write(f"{'COMPREHENSIVE EVALUATION REPORT':^100}\n")
            f.write("=" * 100 + "\n\n")
            
            # Model configuration
            f.write("MODEL CONFIGURATION:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Dataset: {config.dataset_name}\n")
            f.write(f"Backbone: {config.backbone}\n")
            f.write(f"Decoder: {config.decoder}\n")
            f.write(f"Rectify Module: {config.rectify_module}\n")
            f.write(f"Fusion Module: {config.fusion_module}\n")
            f.write(f"Loss Function: {config.criterion}\n")
            f.write(f"Number of Classes: {config.num_classes}\n")
            f.write(f"Number of Eval Images: {self.dataset.get_length()}\n\n")
            
            # Best performing checkpoints
            best_miou_idx = summary_df['mIoU'].idxmax()
            best_pixel_acc_idx = summary_df['Pixel_Acc'].idxmax()
            
            f.write("BEST PERFORMING CHECKPOINTS:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Best mIoU: Epoch {summary_df.loc[best_miou_idx, 'Epoch']} "
                   f"({summary_df.loc[best_miou_idx, 'mIoU']:.2f}%)\n")
            f.write(f"Best Pixel Accuracy: Epoch {summary_df.loc[best_pixel_acc_idx, 'Epoch']} "
                   f"({summary_df.loc[best_pixel_acc_idx, 'Pixel_Acc']:.2f}%)\n\n")
            
            # Summary table
            f.write("SUMMARY METRICS TABLE:\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Epoch':<8} {'mIoU (%)':<10} {'Freq_IoU (%)':<12} {'Mean_Pixel_Acc (%)':<18} "
                   f"{'Pixel_Acc (%)':<14} {'Eval_Time (s)':<12}\n")
            f.write("-" * 80 + "\n")
            
            for _, row in summary_df.iterrows():
                f.write(f"{row['Epoch']:<8} {row['mIoU']:<10.2f} {row['Freq_IoU']:<12.2f} "
                       f"{row['Mean_Pixel_Acc']:<18.2f} {row['Pixel_Acc']:<14.2f} {row['Eval_Time']:<12.1f}\n")
            
            f.write("\n")
            
            # Comprehensive table with all epochs and per-class IoUs
            f.write("COMPREHENSIVE METRICS TABLE (All Checkpoints):\n")
            f.write("-" * 50 + "\n")
            
            # Create header
            header = f"{'Epoch':<8}"
            for i, class_name in enumerate(config.class_names):
                # Truncate long class names for better formatting
                short_name = class_name[:8] if len(class_name) > 8 else class_name
                header += f" {short_name:>8}"
            header += f" {'mIoU':>8} {'mPixAcc':>8}\n"
            f.write(header)
            
            # Create separator line
            separator_length = 8 + (len(config.class_names) * 9) + 17  # Account for spacing
            f.write("-" * separator_length + "\n")
            
            # Write data rows
            for _, row in summary_df.iterrows():
                epoch = int(row['Epoch'])
                epoch_per_class = per_class_df[per_class_df['Epoch'] == epoch]
                
                line = f"{epoch:<8}"
                for class_id in range(len(config.class_names)):
                    class_iou = epoch_per_class[epoch_per_class['Class_ID'] == class_id]['IoU'].iloc[0]
                    line += f" {class_iou:>8.2f}"
                line += f" {row['mIoU']:>8.2f} {row['Mean_Pixel_Acc']:>8.2f}\n"
                f.write(line)
            
            f.write("\n")
            
            # Per-class IoU for best model
            best_epoch = summary_df.loc[best_miou_idx, 'Epoch']
            best_per_class = per_class_df[per_class_df['Epoch'] == best_epoch]
            
            f.write(f"PER-CLASS IoU BREAKDOWN FOR BEST MODEL (Epoch {best_epoch}):\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Class ID':<10} {'Class Name':<25} {'IoU (%)':<10}\n")
            f.write("-" * 45 + "\n")
            
            for _, row in best_per_class.iterrows():
                f.write(f"{row['Class_ID']:<10} {row['Class_Name']:<25} {row['IoU']:<10.2f}\n")
            
            f.write("\n")
            
            # Performance trends
            f.write("PERFORMANCE TRENDS:\n")
            f.write("-" * 50 + "\n")
            initial_miou = summary_df.iloc[0]['mIoU']
            final_miou = summary_df.iloc[-1]['mIoU']
            max_miou = summary_df['mIoU'].max()
            
            f.write(f"Initial mIoU (Epoch {summary_df.iloc[0]['Epoch']}): {initial_miou:.2f}%\n")
            f.write(f"Final mIoU (Epoch {summary_df.iloc[-1]['Epoch']}): {final_miou:.2f}%\n")
            f.write(f"Maximum mIoU: {max_miou:.2f}%\n")
            f.write(f"Improvement over training: {max_miou - initial_miou:.2f}%\n")
            
        logger.info(f"Text report saved to: {report_path}")
        
    def generate_visualizations(self, summary_df, per_class_df):
        """Generate visualization plots"""
        plt.style.use('default')
        
        # 1. Summary metrics over epochs
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Evaluation Metrics Over Training Epochs', fontsize=16, fontweight='bold')
        
        # mIoU
        axes[0, 0].plot(summary_df['Epoch'], summary_df['mIoU'], 'b-o', linewidth=2, markersize=6)
        axes[0, 0].set_title('Mean IoU (%)', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('mIoU (%)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, summary_df['mIoU'].max() * 1.1)
        
        # Pixel Accuracy
        axes[0, 1].plot(summary_df['Epoch'], summary_df['Pixel_Acc'], 'g-o', linewidth=2, markersize=6)
        axes[0, 1].set_title('Pixel Accuracy (%)', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Pixel Accuracy (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Frequency Weighted IoU
        axes[1, 0].plot(summary_df['Epoch'], summary_df['Freq_IoU'], 'r-o', linewidth=2, markersize=6)
        axes[1, 0].set_title('Frequency Weighted IoU (%)', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Freq IoU (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Mean Pixel Accuracy
        axes[1, 1].plot(summary_df['Epoch'], summary_df['Mean_Pixel_Acc'], 'm-o', linewidth=2, markersize=6)
        axes[1, 1].set_title('Mean Pixel Accuracy (%)', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Mean Pixel Acc (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_results_dir, 'summary_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Per-class IoU heatmap
        pivot_df = per_class_df.pivot(index='Class_Name', columns='Epoch', values='IoU')
        
        plt.figure(figsize=(16, 10))
        sns.heatmap(pivot_df, annot=True, fmt='.1f', cmap='RdYlGn', 
                    cbar_kws={'label': 'IoU (%)'}, linewidths=0.5)
        plt.title('Per-Class IoU Across Training Epochs', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Epoch', fontsize=12, fontweight='bold')
        plt.ylabel('Class Name', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_results_dir, 'per_class_iou_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Best vs Worst performing classes
        best_epoch_idx = summary_df['mIoU'].idxmax()
        best_epoch = summary_df.loc[best_epoch_idx, 'Epoch']
        best_class_data = per_class_df[per_class_df['Epoch'] == best_epoch].sort_values('IoU', ascending=False)
        
        plt.figure(figsize=(12, 8))
        colors = ['green' if iou > 50 else 'orange' if iou > 30 else 'red' for iou in best_class_data['IoU']]
        bars = plt.bar(range(len(best_class_data)), best_class_data['IoU'], color=colors, alpha=0.7)
        
        plt.title(f'Per-Class IoU for Best Model (Epoch {best_epoch})', fontsize=14, fontweight='bold')
        plt.xlabel('Class Index', fontsize=12, fontweight='bold')
        plt.ylabel('IoU (%)', fontsize=12, fontweight='bold')
        plt.xticks(range(len(best_class_data)), [f"{row['Class_ID']}" for _, row in best_class_data.iterrows()])
        
        # Add class names as labels on bars
        for i, (_, row) in enumerate(best_class_data.iterrows()):
            plt.text(i, row['IoU'] + 1, f"{row['Class_Name']}", 
                    rotation=45, ha='left', va='bottom', fontsize=8)
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_results_dir, 'best_model_per_class.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to: {self.save_results_dir}")
    
    def generate_basic_report(self, summary_data, per_class_data):
        """Generate basic text report without pandas (fallback)"""
        report_path = os.path.join(self.save_results_dir, 'evaluation_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write(f"{'COMPREHENSIVE EVALUATION REPORT':^100}\n")
            f.write("=" * 100 + "\n\n")
            
            # Model configuration
            f.write("MODEL CONFIGURATION:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Dataset: {config.dataset_name}\n")
            f.write(f"Backbone: {config.backbone}\n")
            f.write(f"Decoder: {config.decoder}\n")
            f.write(f"Number of Classes: {config.num_classes}\n\n")
            
            # Find best performing checkpoints
            best_miou_epoch = max(summary_data, key=lambda x: x['mIoU'])
            best_pixel_acc_epoch = max(summary_data, key=lambda x: x['Pixel_Acc'])
            
            f.write("BEST PERFORMING CHECKPOINTS:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Best mIoU: Epoch {best_miou_epoch['Epoch']} ({best_miou_epoch['mIoU']:.2f}%)\n")
            f.write(f"Best Pixel Accuracy: Epoch {best_pixel_acc_epoch['Epoch']} ({best_pixel_acc_epoch['Pixel_Acc']:.2f}%)\n\n")
            
            # Comprehensive table with all epochs and per-class IoUs
            f.write("COMPREHENSIVE METRICS TABLE (All Checkpoints):\n")
            f.write("-" * 50 + "\n")
            
            # Create header
            header = f"{'Epoch':<8}"
            for i, class_name in enumerate(config.class_names):
                # Truncate long class names for better formatting
                short_name = class_name[:8] if len(class_name) > 8 else class_name
                header += f" {short_name:>8}"
            header += f" {'mIoU':>8} {'mPixAcc':>8}\n"
            f.write(header)
            
            # Create separator line
            separator_length = 8 + (len(config.class_names) * 9) + 17  # Account for spacing
            f.write("-" * separator_length + "\n")
            
            # Write data rows
            for row in sorted(summary_data, key=lambda x: x['Epoch']):
                epoch = row['Epoch']
                epoch_per_class = [item for item in per_class_data if item['Epoch'] == epoch]
                
                line = f"{epoch:<8}"
                for class_id in range(len(config.class_names)):
                    class_iou = next(item['IoU'] for item in epoch_per_class if item['Class_ID'] == class_id)
                    line += f" {class_iou:>8.2f}"
                line += f" {row['mIoU']:>8.2f} {row['Mean_Pixel_Acc']:>8.2f}\n"
                f.write(line)
            
            f.write("\n")
            
            # Summary table
            f.write("SUMMARY METRICS TABLE:\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Epoch':<8} {'mIoU (%)':<10} {'Freq_IoU (%)':<12} {'Mean_Pixel_Acc (%)':<18} "
                   f"{'Pixel_Acc (%)':<14} {'Eval_Time (s)':<12}\n")
            f.write("-" * 80 + "\n")
            
            for row in sorted(summary_data, key=lambda x: x['Epoch']):
                f.write(f"{row['Epoch']:<8} {row['mIoU']:<10.2f} {row['Freq_IoU']:<12.2f} "
                       f"{row['Mean_Pixel_Acc']:<18.2f} {row['Pixel_Acc']:<14.2f} {row['Eval_Time']:<12.1f}\n")
        
        # Save basic CSV files manually
        self.save_basic_csv(summary_data, per_class_data)
        logger.info(f"Basic report saved to: {report_path}")
    
    def save_basic_csv(self, summary_data, per_class_data):
        """Save CSV files without pandas (fallback)"""
        # Save summary metrics
        summary_path = os.path.join(self.save_results_dir, 'summary_metrics.csv')
        with open(summary_path, 'w') as f:
            f.write("Epoch,mIoU,Freq_IoU,Mean_Pixel_Acc,Pixel_Acc,Eval_Time\n")
            for row in sorted(summary_data, key=lambda x: x['Epoch']):
                f.write(f"{row['Epoch']},{row['mIoU']:.4f},{row['Freq_IoU']:.4f},"
                       f"{row['Mean_Pixel_Acc']:.4f},{row['Pixel_Acc']:.4f},{row['Eval_Time']:.2f}\n")
        
        # Save per-class IoU
        per_class_path = os.path.join(self.save_results_dir, 'per_class_iou.csv')
        with open(per_class_path, 'w') as f:
            f.write("Epoch,Class_ID,Class_Name,IoU\n")
            for row in sorted(per_class_data, key=lambda x: (x['Epoch'], x['Class_ID'])):
                f.write(f"{row['Epoch']},{row['Class_ID']},{row['Class_Name']},{row['IoU']:.4f}\n")
        
    def save_detailed_results(self):
        """Save detailed results as JSON for further analysis"""
        detailed_results = {}
        
        for epoch, result in self.all_results.items():
            detailed_results[str(epoch)] = {
                'metrics': {
                    'iou': result['metrics']['iou'].tolist(),
                    'mean_IoU': float(result['metrics']['mean_IoU']),
                    'freq_IoU': float(result['metrics']['freq_IoU']),
                    'mean_pixel_acc': float(result['metrics']['mean_pixel_acc']),
                    'pixel_acc': float(result['metrics']['pixel_acc'])
                },
                'eval_time': result['eval_time'],
                'checkpoint_path': result['checkpoint_path']
            }
        
        with open(os.path.join(self.save_results_dir, 'detailed_results.json'), 'w') as f:
            json.dump(detailed_results, f, indent=2)
            
        logger.info("Detailed results saved as JSON")

def main():
    parser = argparse.ArgumentParser(description='Batch evaluation of all training checkpoints')
    parser.add_argument('-d', '--devices', default='0', type=str,
                        help='GPU devices to use for evaluation')
    parser.add_argument('-v', '--verbose', default=False, action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--save_dir', default='batch_evaluation_results', type=str,
                        help='Directory to save evaluation results')
    parser.add_argument('--epochs', default=None, type=str,
                        help='Specific epochs to evaluate (e.g., "50,100,150" or "50-200")')
    
    args = parser.parse_args()
    
    # Create batch evaluation runner
    runner = BatchEvaluationRunner(
        devices=args.devices,
        verbose=args.verbose,
        save_results_dir=args.save_dir
    )
    
    # Override checkpoint epochs if specified
    if args.epochs:
        if ',' in args.epochs:
            runner.checkpoint_epochs = [int(e.strip()) for e in args.epochs.split(',')]
        elif '-' in args.epochs:
            start, end = map(int, args.epochs.split('-'))
            runner.checkpoint_epochs = list(range(start, end + 1, config.checkpoint_step))
        else:
            runner.checkpoint_epochs = [int(args.epochs)]
    
    # Run batch evaluation
    runner.run_batch_evaluation()
    
    logger.info("Batch evaluation completed successfully!")
    logger.info(f"Results saved in: {args.save_dir}")

if __name__ == "__main__":
    main()
