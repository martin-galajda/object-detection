from utils.image import load_pil_image_from_file
import time


class SpeedTester:

    def __init__(self, path_to_test_image: str):
        _, img_np = load_pil_image_from_file(path_to_test_image)
        print(f'Image width: {img_np.shape[1]}, Image height: {img_np.shape[0]}')
        self.img_np = img_np

    def benchmark_detector(self, detector, num_of_iterations: int):
        start = time.time()
        for i in range(num_of_iterations):
            _ = detector.infer_object_detections_on_loaded_image(self.img_np)

        end = time.time()
        total_time = (end - start)
        seconds_per_inference = total_time / num_of_iterations

        print(f'Took {seconds_per_inference} seconds on average for {detector.name} ')

        return seconds_per_inference

