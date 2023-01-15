# main.py

import argparse

import data_preprocessing.data_loading
import data_preprocessing.data_processing
import data_preprocessing.data_augmentation
import data_preprocessing.data_storage

def main(args):
    # Load the audio data
    if args.url:
        data, sample_rate = data_preprocessing.data_loading.load_from_url(args.url)
    else:
        data, sample_rate = data_preprocessing.data_loading.load_from_file(args.file)
    
    # Preprocess the data
    if args.normalize:
        data = data_preprocessing.data_processing.normalize(data)
    if args.filter:
        data = data_preprocessing.data_processing.apply_filter(data, args.filter)
    if args.extract_features:
        data = data_preprocessing.data_processing.extract_features(data)
    if args.augment:
        data = data_preprocessing.data_augmentation.random_augmentation(data, sample_rate)
    
    # Save the preprocessed data
    if args.save:
        data_preprocessing.data_storage.save_to_file(data, args.save)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help='Path to the audio file to preprocess')
    parser.add_argument('--url', type=str, help='URL of the audio file to preprocess')
    parser.add_argument('--normalize', action='store_true', help='Normalize the audio data')
    parser.add_argument('--filter', type=str, choices=['lowpass', 'highpass', 'bandpass'], help='Apply a filter to the audio data')
    parser.add_argument('--extract_features', action='store_true', help='Extract features from the audio data')
    parser.add_argument('--augment', action='store_true', help='Apply random augmentation to the audio data')
    parser.add_argument('--save', action='store_true', help='Save the preprocessed data')

    args = parser.parse_args()
    main(args)


