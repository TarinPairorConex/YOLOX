# --- Refactored Evaluator Module ---
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict

class VideoFrameEvaluator:
    """Evaluates person-in-bed detection accuracy for video frames"""
    
    def __init__(self):
        """Initialize with ground truth data for various videos"""
        self.frame_ranges = {
            "NTUC-P-57_202506031303_202506031304": [(167, 239)],
            "NTUC-P-57_202506031442_202506031445": [(36, 209), (277, 659)],
            "NTUC-P-57_202506031631_202506031634": [(289, 543)],
            "ruoxuan": [(26, 220)],
            "NTUC-P-57_202506031838_202506031840": [],
            "NTUC-P-57_202506041801_202506041803": [(155, 283)],
            "NTUC-P-57_202506051744_202506051747": [(94, 424)],
            "NTUC-P-57_202506061546_202506061553": [(84, 1512)],
            "NTUC-P-57_202506111624_202506111636": [(28, 1091)],
            "NTUC-P-57_202506111749_202506111750": [(21, 333)],
            "NTUC-P-57_202506111759_202506111802": [(157, 517)],
            "NTUC-P-57_202506111811_202506111813": [(144, 339), (352, 425)],
            "NTUC-P-57_202506111814_202506111816": [(42, 171), (198, 251)],
            "NTUC-P-57_202506121531_202506121533": [(9, 512)],
        }
        
        # Results tracking
        self.results = {}
        self.current_video = None
        self.frame_predictions = []
        self.frame_ground_truth = []
        self.frame_numbers = []
        
    def is_person_on_bed(self, video_path, frame_number):
        """
        Check if a person should be on the bed in the given frame according to ground truth.
        
        Args:
            video_path: Path to the video file
            frame_number: Frame number to check
            
        Returns:
            tuple: (video_path, video_name, ground_truth)
                ground_truth is 1 if person should be on bed, 0 otherwise
        """
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        ranges = self.frame_ranges.get(video_name, [])
        
        for start, end in ranges:
            if start <= frame_number < end:
                return video_path, video_name, 1
                
        return video_path, video_name, 0
    
    def start_evaluation(self, video_path):
        """
        Start evaluation for a new video.
        
        Args:
            video_path: Path to the video file
        """
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.current_video = video_name
        self.frame_predictions = []
        self.frame_ground_truth = []
        self.frame_numbers = []
    
    def add_frame_result(self, frame_number, prediction, ground_truth):
        """
        Add a frame evaluation result.
        
        Args:
            frame_number: Frame number
            prediction: Predicted value (1 if person detected on bed, 0 otherwise)
            ground_truth: Ground truth value (1 if person should be on bed, 0 otherwise)
        """
        if self.current_video is None:
            raise ValueError("Must call start_evaluation before adding frame results")
            
        self.frame_numbers.append(frame_number)
        self.frame_predictions.append(prediction)
        self.frame_ground_truth.append(ground_truth)
    
    def end_evaluation(self):
        """
        End evaluation for the current video and compute metrics.
        
        Returns:
            dict: Evaluation metrics
        """
        if self.current_video is None or not self.frame_predictions:
            return None
            
        # Convert to numpy arrays for easier computation
        predictions = np.array(self.frame_predictions)
        ground_truth = np.array(self.frame_ground_truth)
        
        # Compute metrics
        correct = predictions == ground_truth
        accuracy = np.mean(correct)
        
        # Store results
        self.results[self.current_video] = {
            'frame_numbers': self.frame_numbers,
            'predictions': self.frame_predictions,
            'ground_truth': self.frame_ground_truth,
            'correct': correct.tolist(),
            'accuracy': accuracy
        }
        
        # Reset current evaluation
        result = self.results[self.current_video]
        self.current_video = None
        self.frame_predictions = []
        self.frame_ground_truth = []
        self.frame_numbers = []
        
        return result
    
    def visualize_results(self, video_name=None, save_path=None, show=True):
        """
        Visualize evaluation results for a video.
        
        Args:
            video_name: Name of the video to visualize. If None, visualize all videos.
            save_path: Path to save the visualization. If None, don't save.
            show: Whether to show the visualization.
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        if video_name is not None and video_name not in self.results:
            print(f"No results for video {video_name}")
            return None
            
        if video_name is not None:
            # Visualize a single video
            return self._visualize_single_video(video_name, save_path, show)
        else:
            # Visualize all videos
            return self._visualize_all_videos(save_path, show)
    
    def _visualize_single_video(self, video_name, save_path=None, show=True):
        """Visualize results for a single video"""
        result = self.results[video_name]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot frame-by-frame results
        ax1.plot(result['frame_numbers'], result['ground_truth'], 'g-', label='Ground Truth')
        ax1.plot(result['frame_numbers'], result['predictions'], 'r-', label='Prediction')
        
        # Highlight incorrect frames
        incorrect_frames = [frame for i, frame in enumerate(result['frame_numbers']) 
                           if result['ground_truth'][i] != result['predictions'][i]]
        incorrect_y = [0.5] * len(incorrect_frames)  # Middle of the y-axis
        ax1.scatter(incorrect_frames, incorrect_y, color='red', s=50, marker='x', label='Incorrect')
        
        ax1.set_title(f'Frame-by-Frame Results for {video_name}')
        ax1.set_xlabel('Frame Number')
        ax1.set_ylabel('Person on Bed')
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(['No', 'Yes'])
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy as a bar
        ax2.bar(['Accuracy'], [result['accuracy']], color='blue')
        ax2.set_ylim([0, 1])
        ax2.set_ylabel('Accuracy')
        for i, v in enumerate([result['accuracy']]):
            ax2.text(i, v + 0.01, f'{v:.2%}', ha='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        if show:
            plt.show()
        
        return fig
    
    def _visualize_all_videos(self, save_path=None, show=True):
        """Visualize results for all videos"""
        if not self.results:
            print("No results to visualize")
            return None
            
        # Sort videos by accuracy
        videos = sorted(self.results.keys(), 
                       key=lambda x: self.results[x]['accuracy'])
        
        accuracies = [self.results[video]['accuracy'] for video in videos]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot accuracies as bars
        bars = ax.bar(videos, accuracies, color='blue')
        ax.set_title('Accuracy by Video')
        ax.set_xlabel('Video')
        ax.set_ylabel('Accuracy')
        ax.set_ylim([0, 1])
        
        # Rotate x-axis labels for readability
        plt.xticks(rotation=45, ha='right')
        
        # Add accuracy values on top of bars
        for i, v in enumerate(accuracies):
            ax.text(i, v + 0.01, f'{v:.2%}', ha='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        if show:
            plt.show()
        
        return fig
    
    def visualize_frame_errors(self, video_name, save_path=None, show=True):
        """
        Visualize which frames had errors.
        
        Args:
            video_name: Name of the video to visualize
            save_path: Path to save the visualization. If None, don't save.
            show: Whether to show the visualization.
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        if video_name not in self.results:
            print(f"No results for video {video_name}")
            return None
            
        result = self.results[video_name]
        
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Create a binary array where 1 means correct prediction, 0 means incorrect
        correct_array = np.array([1 if gt == pred else 0 
                                 for gt, pred in zip(result['ground_truth'], 
                                                    result['predictions'])])
        
        # Plot as a heatmap
        cmap = plt.cm.RdYlGn  # Red for errors, green for correct
        ax.imshow(correct_array.reshape(1, -1), cmap=cmap, aspect='auto', vmin=0, vmax=1)
        
        # Set x-axis ticks to frame numbers
        ax.set_xticks(range(len(result['frame_numbers'])))
        ax.set_xticklabels(result['frame_numbers'])
        
        # Only show a subset of frame numbers to avoid overcrowding
        if len(result['frame_numbers']) > 20:
            step = len(result['frame_numbers']) // 20
            ax.set_xticks(range(0, len(result['frame_numbers']), step))
            ax.set_xticklabels(result['frame_numbers'][::step])
        
        ax.set_yticks([])  # No y-axis ticks needed
        ax.set_title(f'Frame Errors for {video_name}')
        ax.set_xlabel('Frame Number')
        
        # Add a colorbar
        cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax)
        cbar.set_ticks([0.25, 0.75])
        cbar.set_ticklabels(['Incorrect', 'Correct'])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        if show:
            plt.show()
        
        return fig
    
    def get_summary(self):
        """
        Get a summary of all evaluation results.
        
        Returns:
            dict: Summary metrics
        """
        if not self.results:
            return None
            
        # Calculate overall metrics
        total_frames = sum(len(result['frame_numbers']) for result in self.results.values())
        total_correct = sum(sum(result['correct']) for result in self.results.values())
        overall_accuracy = total_correct / total_frames if total_frames > 0 else 0
        
        # Get per-video metrics
        video_metrics = {
            video: {
                'frames': len(result['frame_numbers']),
                'correct': sum(result['correct']),
                'accuracy': result['accuracy']
            }
            for video, result in self.results.items()
        }
        
        return {
            'overall': {
                'total_frames': total_frames,
                'total_correct': total_correct,
                'accuracy': overall_accuracy
            },
            'videos': video_metrics
        }
    
    def save_results(self, save_path):
        """
        Save evaluation results to a file.
        
        Args:
            save_path: Path to save the results
        """
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for video, result in self.results.items():
            serializable_results[video] = {
                'frame_numbers': result['frame_numbers'],
                'predictions': result['predictions'],
                'ground_truth': result['ground_truth'],
                'correct': result['correct'],
                'accuracy': float(result['accuracy'])
            }
        
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def load_results(self, load_path):
        """
        Load evaluation results from a file.
        
        Args:
            load_path: Path to load the results from
        """
        import json
        
        with open(load_path, 'r') as f:
            self.results = json.load(f)
