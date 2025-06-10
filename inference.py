import os
import sys
import argparse
import json
import subprocess
import random
import string
import tempfile
import logging
import platform
import shutil # Import shutil

from glob import glob
from tqdm import tqdm
from flask import Flask, request, jsonify
import numpy as np
import scipy, cv2, audio
import torch, face_detection
from models import Wav2Lip

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants - adjust these as needed
CHECKPOINT_PATH = 'checkpoints/wav2lip.pth'  # Replace with your checkpoint file
INPUT_DIR = 'inputs'
OUTPUT_DIR = 'results'
TEMP_DIR = 'temp'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'wav', 'mp4', 'avi'}
CLEANUP_TEMP_FILES = os.environ.get('CLEANUP_TEMP_FILES', 'True').lower() == 'true' # Configurable cleanup

# Initialize Flask application
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # Optional: set max upload size to 100MB

# Initialize CUDA or CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info('Using {} for inference.'.format(device))

# Create argument parser (create it ONCE)
parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--face', type=str,
                    help='Filepath of video/image that contains faces to use', required=False, default=None)
parser.add_argument('--audio', type=str,
                    help='Filepath of video/audio file to use as raw audio source', required=False, default=None)
parser.add_argument('--outfile', type=str, help='Video path to save result.',
                    default=os.path.join(OUTPUT_DIR, 'result_voice.mp4'))

parser.add_argument('--static', type=bool,
                    help='If True, then use only first video frame for inference', default=False)
parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)',
                    default=25., required=False)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0],
                    help='Padding (top, bottom, left, right). Please adjust to include chin at least')

parser.add_argument('--face_det_batch_size', type=int,
                    help='Batch size for face detection', default=16)
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)

parser.add_argument('--resize_factor', default=1, type=int,
            help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1],
                    help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. '
                    'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1],
                    help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
                    'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

parser.add_argument('--rotate', default=False, action='store_true',
                    help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
                    'Use if you get a flipped result, despite feeding a normal looking video')

parser.add_argument('--nosmooth', default=False, action='store_true',
                    help='Prevent smoothing face detections over a short temporal window')

def create_parser():
    parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

    parser.add_argument('--face', type=str,
                        help='Filepath of video/image that contains faces to use', required=False, default=None)
    parser.add_argument('--audio', type=str,
                        help='Filepath of video/audio file to use as raw audio source', required=False, default=None)
    parser.add_argument('--outfile', type=str, help='Video path to save result.',
                        default=os.path.join(OUTPUT_DIR, 'result_voice.mp4'))

    parser.add_argument('--static', type=bool,
                        help='If True, then use only first video frame for inference', default=False)
    parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)',
                        default=25., required=False)

    parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0],
                        help='Padding (top, bottom, left, right). Please adjust to include chin at least')

    parser.add_argument('--face_det_batch_size', type=int,
                        help='Batch size for face detection', default=16)
    parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)

    parser.add_argument('--resize_factor', default=1, type=int,
                help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

    parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1],
                        help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. '
                        'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

    parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1],
                        help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
                        'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

    parser.add_argument('--rotate', default=False, action='store_true',
                        help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
                        'Use if you get a flipped result, despite feeding a normal looking video')

    parser.add_argument('--nosmooth', default=False, action='store_true',
                        help='Prevent smoothing face detections over a short temporal window')

    return parser

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def _load(checkpoint_path):
    try:
        if device == 'cuda':
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        return checkpoint
    except Exception as e:
        logging.error(f"Error loading checkpoint: {e}")
        raise

def load_model(path):
    try:
        model = Wav2Lip()
        logging.info("Load checkpoint from: {}".format(path))
        checkpoint = _load(path)
        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        model.load_state_dict(new_s)

        model = model.to(device)
        return model.eval()
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes

def face_detect(images, args):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                            flip_input=False, device=device)

    batch_size = args.face_det_batch_size

    while 1:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), batch_size)):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1:
                raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            logging.info('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = args.pads
    for rect, image in zip(predictions, images):
        if rect is None:
            cv2.imwrite(os.path.join(TEMP_DIR, 'faulty_frame.jpg'), image)  # check this frame where the face was not detected.
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    return results

def datagen(frames, mels, args):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if args.box[0] == -1:
        if not args.static:
            face_det_results = face_detect(frames, args)  # BGR2RGB for CNN face detection
        else:
            face_det_results = face_detect([frames[0]], args)
    else:
        logging.info('Using the specified bounding box instead of face detection...')
        y1, y2, x1, x2 = args.box
        face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

    for i, m in enumerate(mels):
        idx = 0 if args.static else i % len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (args.img_size, args.img_size))

        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= args.wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, args.img_size // 2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, args.img_size // 2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch


def process_wav2lip(face_file, audio_file, outfile, args):
    """Processes the Wav2Lip inference."""
    try:

        if os.path.isfile(face_file) and face_file.split('.')[-1] in ['jpg', 'png', 'jpeg']:
            full_frames = [cv2.imread(face_file)]
            fps = args.fps
            args.static = True # force static to true

        else:
            video_stream = cv2.VideoCapture(face_file)
            fps = video_stream.get(cv2.CAP_PROP_FPS)

            logging.info('Reading video frames...')

            full_frames = []
            while True:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break
                if args.resize_factor > 1:
                    frame = cv2.resize(frame, (frame.shape[1] // args.resize_factor, frame.shape[0] // args.resize_factor))

                if args.rotate:
                    frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

                y1, y2, x1, x2 = args.crop
                if x2 == -1: x2 = frame.shape[1]
                if y2 == -1: y2 = frame.shape[0]

                frame = frame[y1:y2, x1:x2]

                full_frames.append(frame)

        logging.info("Number of frames available for inference: {}".format(len(full_frames)))

        if not audio_file.endswith('.wav'):
            logging.info('Extracting raw audio...')
            temp_wav = os.path.join(TEMP_DIR, 'temp.wav')
            command = 'ffmpeg -y -i {} -strict -2 {}'.format(audio_file, temp_wav)

            subprocess.call(command, shell=platform.system() != 'Windows') # platform check

            audio_file = temp_wav

        wav = audio.load_wav(audio_file, 16000)
        mel = audio.melspectrogram(wav)
        logging.info("Mel shape: {}".format(mel.shape))

        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

        mel_chunks = []
        mel_idx_multiplier = 80. / fps
        i = 0
        mel_step_size = 16

        while True:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
            i += 1

        logging.info("Length of mel chunks: {}".format(len(mel_chunks)))

        full_frames = full_frames[:len(mel_chunks)]

        batch_size = args.wav2lip_batch_size
        gen = datagen(full_frames.copy(), mel_chunks, args)

        for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, total=int(np.ceil(float(len(mel_chunks)) / batch_size)))):
            if i == 0:
                model = load_model(CHECKPOINT_PATH)
                logging.info("Model loaded")

                frame_h, frame_w = full_frames[0].shape[:-1]
                out = cv2.VideoWriter(os.path.join(TEMP_DIR, 'result.avi'), cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

            with torch.no_grad():
                pred = model(mel_batch, img_batch)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

            for p, f, c in zip(pred, frames, coords):
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

                f[y1:y2, x1:x2] = p
                out.write(f)

        out.release()


        command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(audio_file, os.path.join(TEMP_DIR, 'result.avi'), outfile)
        subprocess.call(command, shell=platform.system() != 'Windows') # platform check


        logging.info(f"Wav2Lip processing complete. Output saved to: {outfile}")
        return outfile  # Return the output file path for API response
    except Exception as e:
        logging.error(f"Error during Wav2Lip processing: {e}")
        raise

# Flask endpoint
@app.route('/process', methods=['GET', 'POST'])
def process_request():
    """Handles the HTTP request to process audio and video."""
    try:
        if request.method == 'GET':
            return jsonify({'message': 'Wav2Lip endpoint is alive. Send a POST request with face and audio.'}), 200

        # Parse arguments
        args = parser.parse_args()
        args.img_size = 96

        # Check if the files are present in the request
        if 'face' not in request.files or 'audio' not in request.files:
            return jsonify({'error': 'Missing face or audio file'}), 400

        face_file = request.files['face']
        audio_file = request.files['audio']

        # Check if the files are allowed
        if not allowed_file(face_file.filename) or not allowed_file(audio_file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        # Save the files to the inputs directory
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(face_file.filename)[1], delete=False, dir=TEMP_DIR) as tmp_face_file:
            face_file.save(tmp_face_file.name)
            face_filepath = tmp_face_file.name
        
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(audio_file.filename)[1], delete=False, dir=TEMP_DIR) as tmp_audio_file:
            audio_file.save(tmp_audio_file.name)
            audio_filepath = tmp_audio_file.name

        args.face = face_filepath
        args.audio = audio_filepath

        # Generate a unique output filename
        output_filename = f"result_{''.join(random.choices(string.ascii_lowercase + string.digits, k=8))}.mp4"
        output_filepath = os.path.join(OUTPUT_DIR, output_filename)
        args.outfile = output_filepath

        # Process Wav2Lip
        result_path = process_wav2lip(args.face, args.audio, args.outfile, args)

        # Clean up temp files
        if CLEANUP_TEMP_FILES:
            try:
                os.remove(face_filepath)
                os.remove(audio_filepath)
                logging.info("Temporary files cleaned up.")
            except OSError as e:
                logging.warning(f"Error cleaning up temporary files: {e}")

        # Return the result
        return jsonify({'result': result_path}), 200

    except Exception as e:
        logging.exception("An error occurred during processing:")
        return jsonify({'error': str(e)}), 500

# Create necessary directories for deployment environments like Cloud Run
def setup_directories():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)

setup_directories()

# For Gunicorn to detect the Flask app
if __name__ != '__main__':
    app.logger.info("Gunicorn loaded Flask app successfully.")

# Cloud Run entrypoint: ensure correct port binding if run as main
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
