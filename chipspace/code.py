# Dependencies and Libraries to add
import numpy as np
import cv2
import base64
import io
from PIL import Image
from inference.core.entities.requests.inference import ObjectDetectionInferenceRequest
from inference.models.yolov5.yolov5_object_detection import YOLOv5ObjectDetection
import warnings
from inference_sdk import InferenceHTTPClient

# Ignore the warnings related to unavailable execution providers
warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime")

# Initialize the model for object detection
try:
    model = YOLOv5ObjectDetection(
        model_id="toddler-final/6",
        api_key="m5baYUlRuefbhMf7DIZx"
    )
    print("Model initialized successfully.")
except Exception as e:
    print(f"An error occurred during model initialization: {e}")
    raise

# Read your input image from your local files
try:
    frame = cv2.imread("Pool_Test_Img.png")
    if frame is None:
        raise ValueError("Failed to read the image. Please check the file path.")
    print("Image read successfully.")
except Exception as e:
    print(f"An error occurred while reading the image: {e}")
    raise

# Convert the frame to base64
try:
    retval, buffer = cv2.imencode('.jpg', frame)
    if not retval:
        raise ValueError("Failed to encode the image to JPEG format.")
    img_str = base64.b64encode(buffer).decode('utf-8')  # Ensure it is a string
    print("Image encoded to base64 successfully.")
except Exception as e:
    print(f"An error occurred during image encoding: {e}")
    raise

# Create inference request using ObjectDetectionInferenceRequest
try:
    request = ObjectDetectionInferenceRequest(
        model_id="toddler-final/6",  # Specify the model_id
        image={
            "type": "base64",
            "value": img_str,
        },
        confidence=0.4,
        iou_threshold=0.5,
        visualization_labels=False,
        visualize_predictions=True
    )
    print("Inference request created successfully.")
except Exception as e:
    print(f"An error occurred while creating the inference request: {e}")
    raise

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="m5baYUlRuefbhMf7DIZx"
)

# G:\My project data\Smart Home Child Protection using Computer Vision/
image_url = "Pool_Test_Img.png"
results = CLIENT.infer(image_url, model_id="toddler-final/6")
print(results)

# Process the results and check for child in unsafe zone
try:
    # Function to convert base64 string back to RGB image
    def stringToRGB(base64_string):
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data))
        opencv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return opencv_img


    # Define pool boundary coordinates
    ##    #Safe Zone
    ##    xmin = 23
    ##    ymin = 84
    ##    xmax = 533
    ##    ymax = 328

    # UnSafe Zone
    xmin = 23
    ymin = 84
    xmax = 733
    ymax = 528

    # Check if child is detected and within unsafe area
    child_detected = False
    for prediction in results['predictions']:
        if "baby" in prediction['class'].lower():
            print("Child detected")
            x1 = int(prediction['x'] - prediction['width'] / 2)
            y1 = int(prediction['y'] - prediction['height'] / 2)
            x2 = int(prediction['x'] + prediction['width'] / 2)
            y2 = int(prediction['y'] + prediction['height'] / 2)

            # Check if child is within the unsafe area
            if x1 > xmin and x2 < xmax and y1 > ymin and y2 < ymax:
                print("Child is in danger! Near Pool Area!")
                # Draw the bounding box on the original image
                cv2.putText(frame, 'Child is in danger! Near Pool Area!', (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (23, 84), (533, 328), (0, 255, 0), 2)
                cv2.imwrite("output_Pool.jpg", frame)
                child_detected = True
                break

    if not child_detected:
        print("No child detected or child outside unsafe area.")
    print("Completed checking for children in dangerous zone.")

    # Display the final image
    cv2.imshow("output_Pool.jpg", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except Exception as e:
    print(f"An error occurred during post-processing: {e}")
    raise

import cv2
import numpy as np

# read image from file path
path = "Fire_Test_Img.jpg"
img = cv2.imread(path)
copy = img.copy()
place_holder = img.copy()

done = False
points = []
current = (0, 0)
prev_current = (0, 0)


# Mouse callbacks
def on_mouse(event, x, y, buttons, user_param):
    global done, points, current, place_holder

    if done:
        return
    if event == cv2.EVENT_MOUSEMOVE:
        # updating the mouse position
        current = (x, y)
    elif event == cv2.EVENT_LBUTTONDOWN:
        # Left click to add a point
        print(x, y)
        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
        points.append([x, y])
        place_holder = img.copy()
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Right click to finish
        print("Boundary complete")
        done = True


cv2.namedWindow("Draw_Boundary")
cv2.setMouseCallback("Draw_Boundary", on_mouse)

while (not done):
    # Keeps drawing new images as we add points
    if (len(points) > 1):
        if (current != prev_current):
            img = place_holder.copy()

        cv2.polylines(img, [np.array(points)], False, (0, 255, 0), 1)
        # To show what the next line would look like
        cv2.line(img, (points[-1][0], points[-1][1]), current, (0, 0, 255))

    # Update the window
    cv2.imshow("Draw_Boundary", img)

    if cv2.waitKey(50) == ord('d'):  # press d(done)
        done = True

# Final drawing
img = copy.copy()

if (len(points) > 0):
    cv2.fillPoly(img, np.array([points]), (255, 0, 0))
    max = np.amax(np.array([points]), axis=1)
    min = np.amin(np.array([points]), axis=1)

    # prints max and min values of the polygon that will be used to draw a rectangle later
    print("xmax:", max[0][0])
    print("ymax:", max[0][1])
    print("xmin:", min[0][0])
    print("ymin:", min[0][1])

# And show it
cv2.imshow("Draw_Boundary", img)
# Waiting for the user to press any key
cv2.waitKey(0)
cv2.destroyWindow("Draw_Boundary")



from time import perf_counter
from typing import Any, List, Tuple, Union

import numpy as np

from inference.core import logger
from inference.core.entities.requests.inference import InferenceRequest
from inference.core.entities.responses.inference import InferenceResponse
from inference.core.models.types import PreprocessReturnMetadata


class BaseInference:
    """General inference class.

    This class provides a basic interface for inference tasks.
    """

    def infer(self, image: Any, **kwargs) -> Any:
        """Runs inference on given data.
        - image:
            can be a BGR numpy array, filepath, InferenceRequestImage, PIL Image, byte-string, etc.
        """
        preproc_image, returned_metadata = self.preprocess(image, **kwargs)
        logger.debug(
            f"Preprocessed input shape: {getattr(preproc_image, 'shape', None)}"
        )
        predicted_arrays = self.predict(preproc_image, **kwargs)
        postprocessed = self.postprocess(predicted_arrays, returned_metadata, **kwargs)

        return postprocessed

    def preprocess(
        self, image: Any, **kwargs
    ) -> Tuple[np.ndarray, PreprocessReturnMetadata]:
        raise NotImplementedError

    def predict(self, img_in: np.ndarray, **kwargs) -> Tuple[np.ndarray, ...]:
        raise NotImplementedError

    def postprocess(
        self,
        predictions: Tuple[np.ndarray, ...],
        preprocess_return_metadata: PreprocessReturnMetadata,
        **kwargs,
    ) -> Any:
        raise NotImplementedError

    def infer_from_request(
        self, request: InferenceRequest
    ) -> Union[InferenceResponse, List[InferenceResponse]]:
        """Runs inference on a request

        Args:
            request (InferenceRequest): The request object.

        Returns:
            Union[CVInferenceResponse, List[CVInferenceResponse]]: The response object(s).

        Raises:
            NotImplementedError: This method must be implemented by a subclass.
        """
        raise NotImplementedError

    def make_response(
        self, *args, **kwargs
    ) -> Union[InferenceResponse, List[InferenceResponse]]:
        """Constructs an object detection response.

        Raises:
            NotImplementedError: This method must be implemented by a subclass.
        """
        raise NotImplementedError


class Model(BaseInference):
    """Base Inference Model (Inherits from BaseInference to define the needed methods)

    This class provides the foundational methods for inference and logging, and can be extended by specific models.

    Methods:
        log(m): Print the given message.
        clear_cache(): Clears any cache if necessary.
    """

    def log(self, m):
        """Prints the given message.

        Args:
            m (str): The message to print.
        """
        print(m)

    def clear_cache(self):
        """Clears any cache if necessary. This method should be implemented in derived classes as needed."""
        pass

    def infer_from_request(
        self,
        request: InferenceRequest,
    ) -> Union[List[InferenceResponse], InferenceResponse]:
        """
        Perform inference based on the details provided in the request, and return the associated responses.
        The function can handle both single and multiple image inference requests. Optionally, it also provides
        a visualization of the predictions if requested.

        Args:
            request (InferenceRequest): The request object containing details for inference, such as the image or
                images to process, any classes to filter by, and whether or not to visualize the predictions.

        Returns:
            Union[List[InferenceResponse], InferenceResponse]: A list of response objects if the request contains
            multiple images, or a single response object if the request contains one image. Each response object
            contains details about the segmented instances, the time taken for inference, and optionally, a visualization.

        Examples:
            >>> request = InferenceRequest(image=my_image, visualize_predictions=True)
            >>> response = infer_from_request(request)
            >>> print(response.time)  # Prints the time taken for inference
            0.125
            >>> print(response.visualization)  # Accesses the visualization of the prediction if available

        Notes:
            - The processing time for each response is included within the response itself.
            - If `visualize_predictions` is set to True in the request, a visualization of the prediction
              is also included in the response.
        """
        t1 = perf_counter()
        responses = self.infer(**request.dict(), return_image_dims=False)
        for response in responses:
            response.time = perf_counter() - t1

        if request.visualize_predictions:
            for response in responses:
                response.visualization = self.draw_predictions(request, response)

        if not isinstance(request.image, list) and len(responses) > 0:
            responses = responses[0]

        return responses

    def make_response(
        self, *args, **kwargs
    ) -> Union[InferenceResponse, List[InferenceResponse]]:
        """Makes an inference response from the given arguments.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            InferenceResponse: The inference response.
        """
        raise NotImplementedError(self.__class__.__name__ + ".make_response")




from io import BytesIO
from time import perf_counter
from typing import Any, List, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from inference.core.entities.requests.inference import ClassificationInferenceRequest
from inference.core.entities.responses.inference import (
    ClassificationInferenceResponse,
    InferenceResponse,
    InferenceResponseImage,
    MultiLabelClassificationInferenceResponse,
)
from inference.core.models.roboflow import OnnxRoboflowInferenceModel
from inference.core.models.types import PreprocessReturnMetadata
from inference.core.models.utils.validate import (
    get_num_classes_from_model_prediction_shape,
)
from inference.core.utils.image_utils import load_image_rgb


class ClassificationBaseOnnxRoboflowInferenceModel(OnnxRoboflowInferenceModel):
    """Base class for ONNX models for Roboflow classification inference.

    Attributes:
        multiclass (bool): Whether the classification is multi-class or not.

    Methods:
        get_infer_bucket_file_list() -> list: Get the list of required files for inference.
        softmax(x): Compute softmax values for a given set of scores.
        infer(request: ClassificationInferenceRequest) -> Union[List[Union[ClassificationInferenceResponse, MultiLabelClassificationInferenceResponse]], Union[ClassificationInferenceResponse, MultiLabelClassificationInferenceResponse]]: Perform inference on a given request and return the response.
        draw_predictions(inference_request, inference_response): Draw prediction visuals on an image.
    """

    task_type = "classification"

    def __init__(self, *args, **kwargs):
        """Initialize the model, setting whether it is multiclass or not."""
        super().__init__(*args, **kwargs)
        self.multiclass = self.environment.get("MULTICLASS", False)

    def draw_predictions(self, inference_request, inference_response):
        """Draw prediction visuals on an image.

        This method overlays the predictions on the input image, including drawing rectangles and text to visualize the predicted classes.

        Args:
            inference_request: The request object containing the image and parameters.
            inference_response: The response object containing the predictions and other details.

        Returns:
            bytes: The bytes of the visualized image in JPEG format.
        """
        image = load_image_rgb(inference_request.image)
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        if isinstance(inference_response.predictions, list):
            prediction = inference_response.predictions[0]
            color = self.colors.get(prediction.class_name, "#4892EA")
            draw.rectangle(
                [0, 0, image.size[1], image.size[0]],
                outline=color,
                width=inference_request.visualization_stroke_width,
            )
            text = f"{prediction.class_id} - {prediction.class_name} {prediction.confidence:.2f}"
            text_size = font.getbbox(text)

            # set button size + 10px margins
            button_size = (text_size[2] + 20, text_size[3] + 20)
            button_img = Image.new("RGBA", button_size, color)
            # put text on button with 10px margins
            button_draw = ImageDraw.Draw(button_img)
            button_draw.text((10, 10), text, font=font, fill=(255, 255, 255, 255))

            # put button on source image in position (0, 0)
            image.paste(button_img, (0, 0))
        else:
            if len(inference_response.predictions) > 0:
                box_color = "#4892EA"
                draw.rectangle(
                    [0, 0, image.size[1], image.size[0]],
                    outline=box_color,
                    width=inference_request.visualization_stroke_width,
                )
            row = 0
            predictions = [
                (cls_name, pred)
                for cls_name, pred in inference_response.predictions.items()
            ]
            predictions = sorted(
                predictions, key=lambda x: x[1].confidence, reverse=True
            )
            for i, (cls_name, pred) in enumerate(predictions):
                color = self.colors.get(cls_name, "#4892EA")
                text = f"{cls_name} {pred.confidence:.2f}"
                text_size = font.getbbox(text)

                # set button size + 10px margins
                button_size = (text_size[2] + 20, text_size[3] + 20)
                button_img = Image.new("RGBA", button_size, color)
                # put text on button with 10px margins
                button_draw = ImageDraw.Draw(button_img)
                button_draw.text((10, 10), text, font=font, fill=(255, 255, 255, 255))

                # put button on source image in position (0, 0)
                image.paste(button_img, (0, row))
                row += button_size[1]

        buffered = BytesIO()
        image = image.convert("RGB")
        image.save(buffered, format="JPEG")
        return buffered.getvalue()

    def get_infer_bucket_file_list(self) -> list:
        """Get the list of required files for inference.

        Returns:
            list: A list of required files for inference, e.g., ["environment.json"].
        """
        return ["environment.json"]

    def infer(
        self,
        image: Any,
        disable_preproc_auto_orient: bool = False,
        disable_preproc_contrast: bool = False,
        disable_preproc_grayscale: bool = False,
        disable_preproc_static_crop: bool = False,
        return_image_dims: bool = False,
        **kwargs,
    ):
        """
        Perform inference on the provided image(s) and return the predictions.

        Args:
            image (Any): The image or list of images to be processed.
                - can be a BGR numpy array, filepath, InferenceRequestImage, PIL Image, byte-string, etc.
            disable_preproc_auto_orient (bool, optional): If true, the auto orient preprocessing step is disabled for this call. Default is False.
            disable_preproc_contrast (bool, optional): If true, the auto contrast preprocessing step is disabled for this call. Default is False.
            disable_preproc_grayscale (bool, optional): If true, the grayscale preprocessing step is disabled for this call. Default is False.
            disable_preproc_static_crop (bool, optional): If true, the static crop preprocessing step is disabled for this call. Default is False.
            return_image_dims (bool, optional): If set to True, the function will also return the dimensions of the image. Defaults to False.
            **kwargs: Additional parameters to customize the inference process.

        Returns:
            Union[List[np.array], np.array, Tuple[List[np.array], List[Tuple[int, int]]], Tuple[np.array, Tuple[int, int]]]:
            If `return_image_dims` is True and a list of images is provided, a tuple containing a list of prediction arrays and a list of image dimensions (width, height) is returned.
            If `return_image_dims` is True and a single image is provided, a tuple containing the prediction array and image dimensions (width, height) is returned.
            If `return_image_dims` is False and a list of images is provided, only the list of prediction arrays is returned.
            If `return_image_dims` is False and a single image is provided, only the prediction array is returned.

        Notes:
            - The input image(s) will be preprocessed (normalized and reshaped) before inference.
            - This function uses an ONNX session to perform inference on the input image(s).
        """
        return super().infer(
            image,
            disable_preproc_auto_orient=disable_preproc_auto_orient,
            disable_preproc_contrast=disable_preproc_contrast,
            disable_preproc_grayscale=disable_preproc_grayscale,
            disable_preproc_static_crop=disable_preproc_static_crop,
            return_image_dims=return_image_dims,
        )

    def postprocess(
        self,
        predictions: Tuple[np.ndarray],
        preprocess_return_metadata: PreprocessReturnMetadata,
        return_image_dims=False,
        **kwargs,
    ) -> Union[ClassificationInferenceResponse, List[ClassificationInferenceResponse]]:
        predictions = predictions[0]
        return self.make_response(
            predictions, preprocess_return_metadata["img_dims"], **kwargs
        )

    def predict(self, img_in: np.ndarray, **kwargs) -> Tuple[np.ndarray]:
        predictions = self.onnx_session.run(None, {self.input_name: img_in})
        return (predictions,)

    def preprocess(
        self, image: Any, **kwargs
    ) -> Tuple[np.ndarray, PreprocessReturnMetadata]:
        if isinstance(image, list):
            imgs_with_dims = [
                self.preproc_image(
                    i,
                    disable_preproc_auto_orient=kwargs.get(
                        "disable_preproc_auto_orient", False
                    ),
                    disable_preproc_contrast=kwargs.get(
                        "disable_preproc_contrast", False
                    ),
                    disable_preproc_grayscale=kwargs.get(
                        "disable_preproc_grayscale", False
                    ),
                    disable_preproc_static_crop=kwargs.get(
                        "disable_preproc_static_crop", False
                    ),
                )
                for i in image
            ]
            imgs, img_dims = zip(*imgs_with_dims)
            img_in = np.concatenate(imgs, axis=0)
        else:
            img_in, img_dims = self.preproc_image(
                image,
                disable_preproc_auto_orient=kwargs.get(
                    "disable_preproc_auto_orient", False
                ),
                disable_preproc_contrast=kwargs.get("disable_preproc_contrast", False),
                disable_preproc_grayscale=kwargs.get(
                    "disable_preproc_grayscale", False
                ),
                disable_preproc_static_crop=kwargs.get(
                    "disable_preproc_static_crop", False
                ),
            )
            img_dims = [img_dims]

        img_in /= 255.0

        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

        img_in = img_in.astype(np.float32)

        img_in[:, 0, :, :] = (img_in[:, 0, :, :] - mean[0]) / std[0]
        img_in[:, 1, :, :] = (img_in[:, 1, :, :] - mean[1]) / std[1]
        img_in[:, 2, :, :] = (img_in[:, 2, :, :] - mean[2]) / std[2]
        return img_in, PreprocessReturnMetadata({"img_dims": img_dims})

    def infer_from_request(
        self,
        request: ClassificationInferenceRequest,
    ) -> Union[List[InferenceResponse], InferenceResponse]:
        """
        Handle an inference request to produce an appropriate response.

        Args:
            request (ClassificationInferenceRequest): The request object encapsulating the image(s) and relevant parameters.

        Returns:
            Union[List[InferenceResponse], InferenceResponse]: The response object(s) containing the predictions, visualization, and other pertinent details. If a list of images was provided, a list of responses is returned. Otherwise, a single response is returned.

        Notes:
            - Starts a timer at the beginning to calculate inference time.
            - Processes the image(s) through the `infer` method.
            - Generates the appropriate response object(s) using `make_response`.
            - Calculates and sets the time taken for inference.
            - If visualization is requested, the predictions are drawn on the image.
        """
        t1 = perf_counter()
        responses = self.infer(**request.dict(), return_image_dims=True)
        for response in responses:
            response.time = perf_counter() - t1

        if request.visualize_predictions:
            for response in responses:
                response.visualization = self.draw_predictions(request, response)

        if not isinstance(request.image, list):
            responses = responses[0]

        return responses

    def make_response(
        self,
        predictions,
        img_dims,
        confidence: float = 0.5,
        **kwargs,
    ) -> Union[ClassificationInferenceResponse, List[ClassificationInferenceResponse]]:
        """
        Create response objects for the given predictions and image dimensions.

        Args:
            predictions (list): List of prediction arrays from the inference process.
            img_dims (list): List of tuples indicating the dimensions (width, height) of each image.
            confidence (float, optional): Confidence threshold for filtering predictions. Defaults to 0.5.
            **kwargs: Additional parameters to influence the response creation process.

        Returns:
            Union[ClassificationInferenceResponse, List[ClassificationInferenceResponse]]: A response object or a list of response objects encapsulating the prediction details.

        Notes:
            - If the model is multiclass, a `MultiLabelClassificationInferenceResponse` is generated for each image.
            - If the model is not multiclass, a `ClassificationInferenceResponse` is generated for each image.
            - Predictions below the confidence threshold are filtered out.
        """
        responses = []
        confidence_threshold = float(confidence)
        for ind, prediction in enumerate(predictions):
            if self.multiclass:
                preds = prediction[0]
                results = dict()
                predicted_classes = []
                for i, o in enumerate(preds):
                    cls_name = self.class_names[i]
                    score = float(o)
                    results[cls_name] = {"confidence": score, "class_id": i}
                    if score > confidence_threshold:
                        predicted_classes.append(cls_name)
                response = MultiLabelClassificationInferenceResponse(
                    image=InferenceResponseImage(
                        width=img_dims[ind][0], height=img_dims[ind][1]
                    ),
                    predicted_classes=predicted_classes,
                    predictions=results,
                )
            else:
                preds = prediction[0]
                preds = self.softmax(preds)
                results = []
                for i, cls_name in enumerate(self.class_names):
                    score = float(preds[i])
                    pred = {
                        "class_id": i,
                        "class": cls_name,
                        "confidence": round(score, 4),
                    }
                    results.append(pred)
                results = sorted(results, key=lambda x: x["confidence"], reverse=True)

                response = ClassificationInferenceResponse(
                    image=InferenceResponseImage(
                        width=img_dims[ind][1], height=img_dims[ind][0]
                    ),
                    predictions=results,
                    top=results[0]["class"],
                    confidence=results[0]["confidence"],
                )
            responses.append(response)

        return responses

    @staticmethod
    def softmax(x):
        """Compute softmax values for each set of scores in x.

        Args:
            x (np.array): The input array containing the scores.

        Returns:
            np.array: The softmax values for each set of scores.
        """
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def get_model_output_shape(self) -> Tuple[int, int, int]:
        test_image = (np.random.rand(1024, 1024, 3) * 255).astype(np.uint8)
        test_image, _ = self.preprocess(test_image)
        output = np.array(self.predict(test_image))
        return output.shape

    def validate_model_classes(self) -> None:
        output_shape = self.get_model_output_shape()
        num_classes = output_shape[3]
        try:
            assert num_classes == self.num_classes
        except AssertionError:
            raise ValueError(
                f"Number of classes in model ({num_classes}) does not match the number of classes in the environment ({self.num_classes})"
            )




DEFAULT_CONFIDENCE = 0.4
DEFAULT_IOU_THRESH = 0.3
DEFAULT_CLASS_AGNOSTIC_NMS = False
DEFAUlT_MAX_DETECTIONS = 300
DEFAULT_MAX_CANDIDATES = 3000



from typing import Any, List, Tuple, Union

import numpy as np

from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    InstanceSegmentationInferenceResponse,
    InstanceSegmentationPrediction,
    Point,
)
from inference.core.exceptions import InvalidMaskDecodeArgument
from inference.core.models.roboflow import OnnxRoboflowInferenceModel
from inference.core.models.types import PreprocessReturnMetadata
from inference.core.models.utils.validate import (
    get_num_classes_from_model_prediction_shape,
)
from inference.core.nms import w_np_non_max_suppression
from inference.core.utils.postprocess import (
    masks2poly,
    post_process_bboxes,
    post_process_polygons,
    process_mask_accurate,
    process_mask_fast,
    process_mask_tradeoff,
)

DEFAULT_CONFIDENCE = 0.4
DEFAULT_IOU_THRESH = 0.3
DEFAULT_CLASS_AGNOSTIC_NMS = False
DEFAUlT_MAX_DETECTIONS = 300
DEFAULT_MAX_CANDIDATES = 3000
DEFAULT_MASK_DECODE_MODE = "accurate"
DEFAULT_TRADEOFF_FACTOR = 0.0

PREDICTIONS_TYPE = List[List[List[float]]]


class InstanceSegmentationBaseOnnxRoboflowInferenceModel(OnnxRoboflowInferenceModel):
    """Roboflow ONNX Instance Segmentation model.

    This class implements an instance segmentation specific inference method
    for ONNX models provided by Roboflow.
    """

    task_type = "instance-segmentation"
    num_masks = 32

    def infer(
        self,
        image: Any,
        class_agnostic_nms: bool = False,
        confidence: float = DEFAULT_CONFIDENCE,
        disable_preproc_auto_orient: bool = False,
        disable_preproc_contrast: bool = False,
        disable_preproc_grayscale: bool = False,
        disable_preproc_static_crop: bool = False,
        iou_threshold: float = DEFAULT_IOU_THRESH,
        mask_decode_mode: str = DEFAULT_MASK_DECODE_MODE,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        max_detections: int = DEFAUlT_MAX_DETECTIONS,
        return_image_dims: bool = False,
        tradeoff_factor: float = DEFAULT_TRADEOFF_FACTOR,
        **kwargs,
    ) -> Union[PREDICTIONS_TYPE, Tuple[PREDICTIONS_TYPE, List[Tuple[int, int]]]]:
        """
        Process an image or list of images for instance segmentation.

        Args:
            image (Any): An image or a list of images for processing.
                - can be a BGR numpy array, filepath, InferenceRequestImage, PIL Image, byte-string, etc.
            class_agnostic_nms (bool, optional): Whether to use class-agnostic non-maximum suppression. Defaults to False.
            confidence (float, optional): Confidence threshold for predictions. Defaults to 0.5.
            iou_threshold (float, optional): IoU threshold for non-maximum suppression. Defaults to 0.5.
            mask_decode_mode (str, optional): Decoding mode for masks. Choices are "accurate", "tradeoff", and "fast". Defaults to "accurate".
            max_candidates (int, optional): Maximum number of candidate detections. Defaults to 3000.
            max_detections (int, optional): Maximum number of detections after non-maximum suppression. Defaults to 300.
            return_image_dims (bool, optional): Whether to return the dimensions of the processed images. Defaults to False.
            tradeoff_factor (float, optional): Tradeoff factor used when `mask_decode_mode` is set to "tradeoff". Must be in [0.0, 1.0]. Defaults to 0.5.
            disable_preproc_auto_orient (bool, optional): If true, the auto orient preprocessing step is disabled for this call. Default is False.
            disable_preproc_contrast (bool, optional): If true, the auto contrast preprocessing step is disabled for this call. Default is False.
            disable_preproc_grayscale (bool, optional): If true, the grayscale preprocessing step is disabled for this call. Default is False.
            disable_preproc_static_crop (bool, optional): If true, the static crop preprocessing step is disabled for this call. Default is False.
            **kwargs: Additional parameters to customize the inference process.

        Returns:
            Union[List[List[List[float]]], Tuple[List[List[List[float]]], List[Tuple[int, int]]]]: The list of predictions, with each prediction being a list of lists. Optionally, also returns the dimensions of the processed images.

        Raises:
            InvalidMaskDecodeArgument: If an invalid `mask_decode_mode` is provided or if the `tradeoff_factor` is outside the allowed range.

        Notes:
            - Processes input images and normalizes them.
            - Makes predictions using the ONNX runtime.
            - Applies non-maximum suppression to the predictions.
            - Decodes the masks according to the specified mode.
        """
        return super().infer(
            image,
            class_agnostic_nms=class_agnostic_nms,
            confidence=confidence,
            disable_preproc_auto_orient=disable_preproc_auto_orient,
            disable_preproc_contrast=disable_preproc_contrast,
            disable_preproc_grayscale=disable_preproc_grayscale,
            disable_preproc_static_crop=disable_preproc_static_crop,
            iou_threshold=iou_threshold,
            mask_decode_mode=mask_decode_mode,
            max_candidates=max_candidates,
            max_detections=max_detections,
            return_image_dims=return_image_dims,
            tradeoff_factor=tradeoff_factor,
        )

    def postprocess(
        self,
        predictions: Tuple[np.ndarray, np.ndarray],
        preprocess_return_metadata: PreprocessReturnMetadata,
        **kwargs,
    ) -> Union[
        InstanceSegmentationInferenceResponse,
        List[InstanceSegmentationInferenceResponse],
    ]:
        predictions, protos = predictions
        predictions = w_np_non_max_suppression(
            predictions,
            conf_thresh=kwargs["confidence"],
            iou_thresh=kwargs["iou_threshold"],
            class_agnostic=kwargs["class_agnostic_nms"],
            max_detections=kwargs["max_detections"],
            max_candidate_detections=kwargs["max_candidates"],
            num_masks=self.num_masks,
        )
        infer_shape = (self.img_size_h, self.img_size_w)
        masks = []
        mask_decode_mode = kwargs["mask_decode_mode"]
        tradeoff_factor = kwargs["tradeoff_factor"]
        img_in_shape = preprocess_return_metadata["im_shape"]

        predictions = [np.array(p) for p in predictions]

        for pred, proto, img_dim in zip(
            predictions, protos, preprocess_return_metadata["img_dims"]
        ):
            if pred.size == 0:
                masks.append([])
                continue
            if mask_decode_mode == "accurate":
                batch_masks = process_mask_accurate(
                    proto, pred[:, 7:], pred[:, :4], img_in_shape[2:]
                )
                output_mask_shape = img_in_shape[2:]
            elif mask_decode_mode == "tradeoff":
                if not 0 <= tradeoff_factor <= 1:
                    raise InvalidMaskDecodeArgument(
                        f"Invalid tradeoff_factor: {tradeoff_factor}. Must be in [0.0, 1.0]"
                    )
                batch_masks = process_mask_tradeoff(
                    proto,
                    pred[:, 7:],
                    pred[:, :4],
                    img_in_shape[2:],
                    tradeoff_factor,
                )
                output_mask_shape = batch_masks.shape[1:]
            elif mask_decode_mode == "fast":
                batch_masks = process_mask_fast(
                    proto, pred[:, 7:], pred[:, :4], img_in_shape[2:]
                )
                output_mask_shape = batch_masks.shape[1:]
            else:
                raise InvalidMaskDecodeArgument(
                    f"Invalid mask_decode_mode: {mask_decode_mode}. Must be one of ['accurate', 'fast', 'tradeoff']"
                )
            polys = masks2poly(batch_masks)
            pred[:, :4] = post_process_bboxes(
                [pred[:, :4]],
                infer_shape,
                [img_dim],
                self.preproc,
                resize_method=self.resize_method,
                disable_preproc_static_crop=preprocess_return_metadata[
                    "disable_preproc_static_crop"
                ],
            )[0]
            polys = post_process_polygons(
                img_dim,
                polys,
                output_mask_shape,
                self.preproc,
                resize_method=self.resize_method,
            )
            masks.append(polys)
        return self.make_response(
            predictions, masks, preprocess_return_metadata["img_dims"], **kwargs
        )

    def preprocess(
        self, image: Any, **kwargs
    ) -> Tuple[np.ndarray, PreprocessReturnMetadata]:
        img_in, img_dims = self.load_image(
            image,
            disable_preproc_auto_orient=kwargs.get("disable_preproc_auto_orient"),
            disable_preproc_contrast=kwargs.get("disable_preproc_contrast"),
            disable_preproc_grayscale=kwargs.get("disable_preproc_grayscale"),
            disable_preproc_static_crop=kwargs.get("disable_preproc_static_crop"),
        )

        img_in /= 255.0
        return img_in, PreprocessReturnMetadata(
            {
                "img_dims": img_dims,
                "im_shape": img_in.shape,
                "disable_preproc_static_crop": kwargs.get(
                    "disable_preproc_static_crop"
                ),
            }
        )

    def make_response(
        self,
        predictions: List[List[List[float]]],
        masks: List[List[List[float]]],
        img_dims: List[Tuple[int, int]],
        class_filter: List[str] = [],
        **kwargs,
    ) -> Union[
        InstanceSegmentationInferenceResponse,
        List[InstanceSegmentationInferenceResponse],
    ]:
        """
        Create instance segmentation inference response objects for the provided predictions and masks.

        Args:
            predictions (List[List[List[float]]]): List of prediction data, one for each image.
            masks (List[List[List[float]]]): List of masks corresponding to the predictions.
            img_dims (List[Tuple[int, int]]): List of image dimensions corresponding to the processed images.
            class_filter (List[str], optional): List of class names to filter predictions by. Defaults to an empty list (no filtering).

        Returns:
            Union[InstanceSegmentationInferenceResponse, List[InstanceSegmentationInferenceResponse]]: A single instance segmentation response or a list of instance segmentation responses based on the number of processed images.

        Notes:
            - For each image, constructs an `InstanceSegmentationInferenceResponse` object.
            - Each response contains a list of `InstanceSegmentationPrediction` objects.
        """
        responses = []
        for ind, (batch_predictions, batch_masks) in enumerate(zip(predictions, masks)):
            predictions = []
            for pred, mask in zip(batch_predictions, batch_masks):
                if class_filter and self.class_names[int(pred[6])] in class_filter:
                    # TODO: logger.debug
                    continue
                # Passing args as a dictionary here since one of the args is 'class' (a protected term in Python)
                predictions.append(
                    InstanceSegmentationPrediction(
                        **{
                            "x": pred[0] + (pred[2] - pred[0]) / 2,
                            "y": pred[1] + (pred[3] - pred[1]) / 2,
                            "width": pred[2] - pred[0],
                            "height": pred[3] - pred[1],
                            "points": [Point(x=point[0], y=point[1]) for point in mask],
                            "confidence": pred[4],
                            "class": self.class_names[int(pred[6])],
                            "class_id": int(pred[6]),
                        }
                    )
                )
            response = InstanceSegmentationInferenceResponse(
                predictions=predictions,
                image=InferenceResponseImage(
                    width=img_dims[ind][1], height=img_dims[ind][0]
                ),
            )
            responses.append(response)
        return responses

    def predict(self, img_in: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Runs inference on the ONNX model.

        Args:
            img_in (np.ndarray): The preprocessed image(s) to run inference on.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The ONNX model predictions and the ONNX model protos.

        Raises:
            NotImplementedError: This method must be implemented by a subclass.
        """
        raise NotImplementedError("predict must be implemented by a subclass")

    def validate_model_classes(self) -> None:
        output_shape = self.get_model_output_shape()
        num_classes = get_num_classes_from_model_prediction_shape(
            output_shape[2], masks=self.num_masks
        )
        try:
            assert num_classes == self.num_classes
        except AssertionError:
            raise ValueError(
                f"Number of classes in model ({num_classes}) does not match the number of classes in the environment ({self.num_classes})"
            )



from typing import List, Optional, Tuple

import numpy as np

from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    Keypoint,
    KeypointsDetectionInferenceResponse,
    KeypointsPrediction,
)
from inference.core.exceptions import ModelArtefactError
from inference.core.models.object_detection_base import (
    ObjectDetectionBaseOnnxRoboflowInferenceModel,
)
from inference.core.models.types import PreprocessReturnMetadata
from inference.core.models.utils.keypoints import model_keypoints_to_response
from inference.core.models.utils.validate import (
    get_num_classes_from_model_prediction_shape,
)
from inference.core.nms import w_np_non_max_suppression
from inference.core.utils.postprocess import post_process_bboxes, post_process_keypoints

DEFAULT_CONFIDENCE = 0.4
DEFAULT_IOU_THRESH = 0.3
DEFAULT_CLASS_AGNOSTIC_NMS = False
DEFAUlT_MAX_DETECTIONS = 300
DEFAULT_MAX_CANDIDATES = 3000


class KeypointsDetectionBaseOnnxRoboflowInferenceModel(
    ObjectDetectionBaseOnnxRoboflowInferenceModel
):
    """Roboflow ONNX Object detection model. This class implements an object detection specific infer method."""

    task_type = "keypoint-detection"

    def __init__(self, model_id: str, *args, **kwargs):
        super().__init__(model_id, *args, **kwargs)

    def get_infer_bucket_file_list(self) -> list:
        """Returns the list of files to be downloaded from the inference bucket for ONNX model.

        Returns:
            list: A list of filenames specific to ONNX models.
        """
        return ["environment.json", "class_names.txt", "keypoints_metadata.json"]

    def postprocess(
        self,
        predictions: Tuple[np.ndarray],
        preproc_return_metadata: PreprocessReturnMetadata,
        class_agnostic_nms=DEFAULT_CLASS_AGNOSTIC_NMS,
        confidence: float = DEFAULT_CONFIDENCE,
        iou_threshold: float = DEFAULT_IOU_THRESH,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        max_detections: int = DEFAUlT_MAX_DETECTIONS,
        return_image_dims: bool = False,
        **kwargs,
    ) -> List[KeypointsDetectionInferenceResponse]:
        """Postprocesses the object detection predictions.

        Args:
            predictions (np.ndarray): Raw predictions from the model.
            img_dims (List[Tuple[int, int]]): Dimensions of the images.
            class_agnostic_nms (bool): Whether to apply class-agnostic non-max suppression. Default is False.
            confidence (float): Confidence threshold for filtering detections. Default is 0.5.
            iou_threshold (float): IoU threshold for non-max suppression. Default is 0.5.
            max_candidates (int): Maximum number of candidate detections. Default is 3000.
            max_detections (int): Maximum number of final detections. Default is 300.

        Returns:
            List[KeypointsDetectionInferenceResponse]: The post-processed predictions.
        """
        predictions = predictions[0]
        number_of_classes = len(self.get_class_names)
        num_masks = predictions.shape[2] - 5 - number_of_classes
        predictions = w_np_non_max_suppression(
            predictions,
            conf_thresh=confidence,
            iou_thresh=iou_threshold,
            class_agnostic=class_agnostic_nms,
            max_detections=max_detections,
            max_candidate_detections=max_candidates,
            num_masks=num_masks,
        )

        infer_shape = (self.img_size_h, self.img_size_w)
        img_dims = preproc_return_metadata["img_dims"]
        predictions = post_process_bboxes(
            predictions=predictions,
            infer_shape=infer_shape,
            img_dims=img_dims,
            preproc=self.preproc,
            resize_method=self.resize_method,
            disable_preproc_static_crop=preproc_return_metadata[
                "disable_preproc_static_crop"
            ],
        )
        predictions = post_process_keypoints(
            predictions=predictions,
            keypoints_start_index=-num_masks,
            infer_shape=infer_shape,
            img_dims=img_dims,
            preproc=self.preproc,
            resize_method=self.resize_method,
            disable_preproc_static_crop=preproc_return_metadata[
                "disable_preproc_static_crop"
            ],
        )
        return self.make_response(predictions, img_dims, **kwargs)

    def make_response(
        self,
        predictions: List[List[float]],
        img_dims: List[Tuple[int, int]],
        class_filter: Optional[List[str]] = None,
        *args,
        **kwargs,
    ) -> List[KeypointsDetectionInferenceResponse]:
        """Constructs object detection response objects based on predictions.

        Args:
            predictions (List[List[float]]): The list of predictions.
            img_dims (List[Tuple[int, int]]): Dimensions of the images.
            class_filter (Optional[List[str]]): A list of class names to filter, if provided.

        Returns:
            List[KeypointsDetectionInferenceResponse]: A list of response objects containing keypoints detection predictions.
        """
        if isinstance(img_dims, dict) and "img_dims" in img_dims:
            img_dims = img_dims["img_dims"]
        keypoint_confidence_threshold = 0.0
        if "request" in kwargs:
            keypoint_confidence_threshold = kwargs["request"].keypoint_confidence
        responses = [
            KeypointsDetectionInferenceResponse(
                predictions=[
                    KeypointsPrediction(
                        # Passing args as a dictionary here since one of the args is 'class' (a protected term in Python)
                        **{
                            "x": (pred[0] + pred[2]) / 2,
                            "y": (pred[1] + pred[3]) / 2,
                            "width": pred[2] - pred[0],
                            "height": pred[3] - pred[1],
                            "confidence": pred[4],
                            "class": self.class_names[int(pred[6])],
                            "class_id": int(pred[6]),
                            "keypoints": model_keypoints_to_response(
                                keypoints_metadata=self.keypoints_metadata,
                                keypoints=pred[7:],
                                predicted_object_class_id=int(pred[6]),
                                keypoint_confidence_threshold=keypoint_confidence_threshold,
                            ),
                        }
                    )
                    for pred in batch_predictions
                    if not class_filter
                    or self.class_names[int(pred[6])] in class_filter
                ],
                image=InferenceResponseImage(
                    width=img_dims[ind][1], height=img_dims[ind][0]
                ),
            )
            for ind, batch_predictions in enumerate(predictions)
        ]
        return responses

    def keypoints_count(self) -> int:
        raise NotImplementedError

    def validate_model_classes(self) -> None:
        num_keypoints = self.keypoints_count()
        output_shape = self.get_model_output_shape()
        num_classes = get_num_classes_from_model_prediction_shape(
            len_prediction=output_shape[2], keypoints=num_keypoints
        )
        if num_classes != self.num_classes:
            raise ValueError(
                f"Number of classes in model ({num_classes}) does not match the number of classes in the environment ({self.num_classes})"
            )



from typing import Any, List, Optional, Tuple, Union

import numpy as np

from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
)
from inference.core.env import FIX_BATCH_SIZE, MAX_BATCH_SIZE
from inference.core.logger import logger
from inference.core.models.defaults import (
    DEFAULT_CLASS_AGNOSTIC_NMS,
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU_THRESH,
    DEFAULT_MAX_CANDIDATES,
    DEFAUlT_MAX_DETECTIONS,
)
from inference.core.models.roboflow import OnnxRoboflowInferenceModel
from inference.core.models.types import PreprocessReturnMetadata
from inference.core.models.utils.validate import (
    get_num_classes_from_model_prediction_shape,
)
from inference.core.nms import w_np_non_max_suppression
from inference.core.utils.postprocess import post_process_bboxes


class ObjectDetectionBaseOnnxRoboflowInferenceModel(OnnxRoboflowInferenceModel):
    """Roboflow ONNX Object detection model. This class implements an object detection specific infer method."""

    task_type = "object-detection"
    box_format = "xywh"

    def infer(
        self,
        image: Any,
        class_agnostic_nms: bool = DEFAULT_CLASS_AGNOSTIC_NMS,
        confidence: float = DEFAULT_CONFIDENCE,
        disable_preproc_auto_orient: bool = False,
        disable_preproc_contrast: bool = False,
        disable_preproc_grayscale: bool = False,
        disable_preproc_static_crop: bool = False,
        iou_threshold: float = DEFAULT_IOU_THRESH,
        fix_batch_size: bool = False,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        max_detections: int = DEFAUlT_MAX_DETECTIONS,
        return_image_dims: bool = False,
        **kwargs,
    ) -> Any:
        """
        Runs object detection inference on one or multiple images and returns the detections.

        Args:
            image (Any): The input image or a list of images to process.
                - can be a BGR numpy array, filepath, InferenceRequestImage, PIL Image, byte-string, etc.
            class_agnostic_nms (bool, optional): Whether to use class-agnostic non-maximum suppression. Defaults to False.
            confidence (float, optional): Confidence threshold for predictions. Defaults to 0.5.
            iou_threshold (float, optional): IoU threshold for non-maximum suppression. Defaults to 0.5.
            fix_batch_size (bool, optional): If True, fix the batch size for predictions. Useful when the model requires a fixed batch size. Defaults to False.
            max_candidates (int, optional): Maximum number of candidate detections. Defaults to 3000.
            max_detections (int, optional): Maximum number of detections after non-maximum suppression. Defaults to 300.
            return_image_dims (bool, optional): Whether to return the dimensions of the processed images along with the predictions. Defaults to False.
            disable_preproc_auto_orient (bool, optional): If true, the auto orient preprocessing step is disabled for this call. Default is False.
            disable_preproc_contrast (bool, optional): If true, the auto contrast preprocessing step is disabled for this call. Default is False.
            disable_preproc_grayscale (bool, optional): If true, the grayscale preprocessing step is disabled for this call. Default is False.
            disable_preproc_static_crop (bool, optional): If true, the static crop preprocessing step is disabled for this call. Default is False.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Union[List[ObjectDetectionInferenceResponse], ObjectDetectionInferenceResponse]: One or multiple object detection inference responses based on the number of processed images. Each response contains a list of predictions. If `return_image_dims` is True, it will return a tuple with predictions and image dimensions.

        Raises:
            ValueError: If batching is not enabled for the model and more than one image is passed for processing.
        """
        return super().infer(
            image,
            class_agnostic_nms=class_agnostic_nms,
            confidence=confidence,
            disable_preproc_auto_orient=disable_preproc_auto_orient,
            disable_preproc_contrast=disable_preproc_contrast,
            disable_preproc_grayscale=disable_preproc_grayscale,
            disable_preproc_static_crop=disable_preproc_static_crop,
            iou_threshold=iou_threshold,
            fix_batch_size=fix_batch_size,
            max_candidates=max_candidates,
            max_detections=max_detections,
            return_image_dims=return_image_dims,
            **kwargs,
        )

    def make_response(
        self,
        predictions: List[List[float]],
        img_dims: List[Tuple[int, int]],
        class_filter: Optional[List[str]] = None,
        *args,
        **kwargs,
    ) -> List[ObjectDetectionInferenceResponse]:
        """Constructs object detection response objects based on predictions.

        Args:
            predictions (List[List[float]]): The list of predictions.
            img_dims (List[Tuple[int, int]]): Dimensions of the images.
            class_filter (Optional[List[str]]): A list of class names to filter, if provided.

        Returns:
            List[ObjectDetectionInferenceResponse]: A list of response objects containing object detection predictions.
        """

        if isinstance(img_dims, dict) and "img_dims" in img_dims:
            img_dims = img_dims["img_dims"]

        predictions = predictions[
            : len(img_dims)
        ]  # If the batch size was fixed we have empty preds at the end
        responses = [
            ObjectDetectionInferenceResponse(
                predictions=[
                    ObjectDetectionPrediction(
                        # Passing args as a dictionary here since one of the args is 'class' (a protected term in Python)
                        **{
                            "x": (pred[0] + pred[2]) / 2,
                            "y": (pred[1] + pred[3]) / 2,
                            "width": pred[2] - pred[0],
                            "height": pred[3] - pred[1],
                            "confidence": pred[4],
                            "class": self.class_names[int(pred[6])],
                            "class_id": int(pred[6]),
                        }
                    )
                    for pred in batch_predictions
                    if not class_filter
                    or self.class_names[int(pred[6])] in class_filter
                ],
                image=InferenceResponseImage(
                    width=img_dims[ind][1], height=img_dims[ind][0]
                ),
            )
            for ind, batch_predictions in enumerate(predictions)
        ]
        return responses

    def postprocess(
        self,
        predictions: Tuple[np.ndarray, ...],
        preproc_return_metadata: PreprocessReturnMetadata,
        class_agnostic_nms=DEFAULT_CLASS_AGNOSTIC_NMS,
        confidence: float = DEFAULT_CONFIDENCE,
        iou_threshold: float = DEFAULT_IOU_THRESH,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        max_detections: int = DEFAUlT_MAX_DETECTIONS,
        return_image_dims: bool = False,
        **kwargs,
    ) -> List[ObjectDetectionInferenceResponse]:
        """Postprocesses the object detection predictions.

        Args:
            predictions (np.ndarray): Raw predictions from the model.
            img_dims (List[Tuple[int, int]]): Dimensions of the images.
            class_agnostic_nms (bool): Whether to apply class-agnostic non-max suppression. Default is False.
            confidence (float): Confidence threshold for filtering detections. Default is 0.5.
            iou_threshold (float): IoU threshold for non-max suppression. Default is 0.5.
            max_candidates (int): Maximum number of candidate detections. Default is 3000.
            max_detections (int): Maximum number of final detections. Default is 300.

        Returns:
            List[ObjectDetectionInferenceResponse]: The post-processed predictions.
        """
        predictions = predictions[0]

        predictions = w_np_non_max_suppression(
            predictions,
            conf_thresh=confidence,
            iou_thresh=iou_threshold,
            class_agnostic=class_agnostic_nms,
            max_detections=max_detections,
            max_candidate_detections=max_candidates,
            box_format=self.box_format,
        )

        infer_shape = (self.img_size_h, self.img_size_w)
        img_dims = preproc_return_metadata["img_dims"]
        predictions = post_process_bboxes(
            predictions,
            infer_shape,
            img_dims,
            self.preproc,
            resize_method=self.resize_method,
            disable_preproc_static_crop=preproc_return_metadata[
                "disable_preproc_static_crop"
            ],
        )
        return self.make_response(predictions, img_dims, **kwargs)

    def preprocess(
        self,
        image: Any,
        disable_preproc_auto_orient: bool = False,
        disable_preproc_contrast: bool = False,
        disable_preproc_grayscale: bool = False,
        disable_preproc_static_crop: bool = False,
        fix_batch_size: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, PreprocessReturnMetadata]:
        """Preprocesses an object detection inference request.

        Args:
            request (ObjectDetectionInferenceRequest): The request object containing images.

        Returns:
            Tuple[np.ndarray, List[Tuple[int, int]]]: Preprocessed image inputs and corresponding dimensions.
        """
        img_in, img_dims = self.load_image(
            image,
            disable_preproc_auto_orient=disable_preproc_auto_orient,
            disable_preproc_contrast=disable_preproc_contrast,
            disable_preproc_grayscale=disable_preproc_grayscale,
            disable_preproc_static_crop=disable_preproc_static_crop,
        )

        img_in /= 255.0

        if self.batching_enabled:
            batch_padding = 0
            if FIX_BATCH_SIZE or fix_batch_size:
                if MAX_BATCH_SIZE == float("inf"):
                    logger.warn(
                        "Requested fix_batch_size but MAX_BATCH_SIZE is not set. Using dynamic batching."
                    )
                    batch_padding = 0
                else:
                    batch_padding = MAX_BATCH_SIZE - img_in.shape[0]
            if batch_padding < 0:
                raise ValueError(
                    f"Requested fix_batch_size but passed in {img_in.shape[0]} images "
                    f"when the model's batch size is {MAX_BATCH_SIZE}\n"
                    f"Consider turning off fix_batch_size, changing `MAX_BATCH_SIZE` in"
                    f"your inference server config, or passing at most {MAX_BATCH_SIZE} images at a time"
                )
            width_remainder = img_in.shape[2] % 32
            height_remainder = img_in.shape[3] % 32
            if width_remainder > 0:
                width_padding = 32 - (img_in.shape[2] % 32)
            else:
                width_padding = 0
            if height_remainder > 0:
                height_padding = 32 - (img_in.shape[3] % 32)
            else:
                height_padding = 0
            img_in = np.pad(
                img_in,
                ((0, batch_padding), (0, 0), (0, width_padding), (0, height_padding)),
                "constant",
            )

        return img_in, PreprocessReturnMetadata(
            {
                "img_dims": img_dims,
                "disable_preproc_static_crop": disable_preproc_static_crop,
            }
        )

    def predict(self, img_in: np.ndarray, **kwargs) -> Tuple[np.ndarray]:
        """Runs inference on the ONNX model.

        Args:
            img_in (np.ndarray): The preprocessed image(s) to run inference on.

        Returns:
            Tuple[np.ndarray]: The ONNX model predictions.

        Raises:
            NotImplementedError: This method must be implemented by a subclass.
        """
        raise NotImplementedError("predict must be implemented by a subclass")

    def validate_model_classes(self) -> None:
        output_shape = self.get_model_output_shape()
        num_classes = get_num_classes_from_model_prediction_shape(
            output_shape[2], masks=0
        )
        try:
            assert num_classes == self.num_classes
        except AssertionError:
            raise ValueError(
                f"Number of classes in model ({num_classes}) does not match the number of classes in the environment ({self.num_classes})"
            )



import itertools
import json
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import onnxruntime
from PIL import Image

from inference.core.cache import cache
from inference.core.cache.model_artifacts import (
    are_all_files_cached,
    clear_cache,
    get_cache_dir,
    get_cache_file_path,
    initialise_cache,
    load_json_from_cache,
    load_text_file_from_cache,
    save_bytes_in_cache,
    save_json_in_cache,
    save_text_lines_in_cache,
)
from inference.core.devices.utils import GLOBAL_DEVICE_ID
from inference.core.entities.requests.inference import (
    InferenceRequest,
    InferenceRequestImage,
)
from inference.core.entities.responses.inference import InferenceResponse
from inference.core.env import (
    API_KEY,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    CORE_MODEL_BUCKET,
    DISABLE_PREPROC_AUTO_ORIENT,
    INFER_BUCKET,
    LAMBDA,
    MAX_BATCH_SIZE,
    MODEL_CACHE_DIR,
    MODEL_VALIDATION_DISABLED,
    ONNXRUNTIME_EXECUTION_PROVIDERS,
    REQUIRED_ONNX_PROVIDERS,
    TENSORRT_CACHE_PATH,
)
from inference.core.exceptions import ModelArtefactError, OnnxProviderNotAvailable
from inference.core.logger import logger
from inference.core.models.base import Model
from inference.core.models.utils.batching import create_batches
from inference.core.models.utils.onnx import has_trt
from inference.core.roboflow_api import (
    ModelEndpointType,
    get_from_url,
    get_roboflow_model_data,
)
from inference.core.utils.image_utils import load_image
from inference.core.utils.onnx import get_onnxruntime_execution_providers
from inference.core.utils.preprocess import letterbox_image, prepare
from inference.core.utils.visualisation import draw_detection_predictions
from inference.models.aliases import resolve_roboflow_model_alias

NUM_S3_RETRY = 5
SLEEP_SECONDS_BETWEEN_RETRIES = 3
MODEL_METADATA_CACHE_EXPIRATION_TIMEOUT = 3600  # 1 hour

S3_CLIENT = None
if AWS_ACCESS_KEY_ID and AWS_ACCESS_KEY_ID:
    try:
        import boto3
        from botocore.config import Config

        from inference.core.utils.s3 import download_s3_files_to_directory

        config = Config(retries={"max_attempts": NUM_S3_RETRY, "mode": "standard"})
        S3_CLIENT = boto3.client("s3", config=config)
    except:
        logger.debug("Error loading boto3")
        pass

DEFAULT_COLOR_PALETTE = [
    "#4892EA",
    "#00EEC3",
    "#FE4EF0",
    "#F4004E",
    "#FA7200",
    "#EEEE17",
    "#90FF00",
    "#78C1D2",
    "#8C29FF",
]


class RoboflowInferenceModel(Model):
    """Base Roboflow inference model."""

    def __init__(
        self,
        model_id: str,
        cache_dir_root=MODEL_CACHE_DIR,
        api_key=None,
        load_weights=True,
    ):
        """
        Initialize the RoboflowInferenceModel object.

        Args:
            model_id (str): The unique identifier for the model.
            cache_dir_root (str, optional): The root directory for the cache. Defaults to MODEL_CACHE_DIR.
            api_key (str, optional): API key for authentication. Defaults to None.
        """
        super().__init__()
        self.load_weights = load_weights
        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}
        self.api_key = api_key if api_key else API_KEY
        model_id = resolve_roboflow_model_alias(model_id=model_id)
        self.dataset_id, self.version_id = model_id.split("/")
        self.endpoint = model_id
        self.device_id = GLOBAL_DEVICE_ID
        self.cache_dir = os.path.join(cache_dir_root, self.endpoint)
        self.keypoints_metadata: Optional[dict] = None
        initialise_cache(model_id=self.endpoint)

    def cache_file(self, f: str) -> str:
        """Get the cache file path for a given file.

        Args:
            f (str): Filename.

        Returns:
            str: Full path to the cached file.
        """
        return get_cache_file_path(file=f, model_id=self.endpoint)

    def clear_cache(self) -> None:
        """Clear the cache directory."""
        clear_cache(model_id=self.endpoint)

    def draw_predictions(
        self,
        inference_request: InferenceRequest,
        inference_response: InferenceResponse,
    ) -> bytes:
        """Draw predictions from an inference response onto the original image provided by an inference request

        Args:
            inference_request (ObjectDetectionInferenceRequest): The inference request containing the image on which to draw predictions
            inference_response (ObjectDetectionInferenceResponse): The inference response containing predictions to be drawn

        Returns:
            str: A base64 encoded image string
        """
        return draw_detection_predictions(
            inference_request=inference_request,
            inference_response=inference_response,
            colors=self.colors,
        )

    @property
    def get_class_names(self):
        return self.class_names

    def get_device_id(self) -> str:
        """
        Get the device identifier on which the model is deployed.

        Returns:
            str: Device identifier.
        """
        return self.device_id

    def get_infer_bucket_file_list(self) -> List[str]:
        """Get a list of inference bucket files.

        Raises:
            NotImplementedError: If the method is not implemented.

        Returns:
            List[str]: A list of inference bucket files.
        """
        raise NotImplementedError(
            self.__class__.__name__ + ".get_infer_bucket_file_list"
        )

    @property
    def cache_key(self):
        return f"metadata:{self.endpoint}"

    @staticmethod
    def model_metadata_from_memcache_endpoint(endpoint):
        model_metadata = cache.get(f"metadata:{endpoint}")
        return model_metadata

    def model_metadata_from_memcache(self):
        model_metadata = cache.get(self.cache_key)
        return model_metadata

    def write_model_metadata_to_memcache(self, metadata):
        cache.set(
            self.cache_key, metadata, expire=MODEL_METADATA_CACHE_EXPIRATION_TIMEOUT
        )

    @property
    def has_model_metadata(self):
        return self.model_metadata_from_memcache() is not None

    def get_model_artifacts(self) -> None:
        """Fetch or load the model artifacts.

        Downloads the model artifacts from S3 or the Roboflow API if they are not already cached.
        """
        self.cache_model_artefacts()
        self.load_model_artifacts_from_cache()

    def cache_model_artefacts(self) -> None:
        infer_bucket_files = self.get_all_required_infer_bucket_file()
        if are_all_files_cached(files=infer_bucket_files, model_id=self.endpoint):
            return None
        if is_model_artefacts_bucket_available():
            self.download_model_artefacts_from_s3()
            return None
        self.download_model_artifacts_from_roboflow_api()

    def get_all_required_infer_bucket_file(self) -> List[str]:
        infer_bucket_files = self.get_infer_bucket_file_list()
        infer_bucket_files.append(self.weights_file)
        logger.debug(f"List of files required to load model: {infer_bucket_files}")
        return [f for f in infer_bucket_files if f is not None]

    def download_model_artefacts_from_s3(self) -> None:
        try:
            logger.debug("Downloading model artifacts from S3")
            infer_bucket_files = self.get_all_required_infer_bucket_file()
            cache_directory = get_cache_dir()
            s3_keys = [f"{self.endpoint}/{file}" for file in infer_bucket_files]
            download_s3_files_to_directory(
                bucket=self.model_artifact_bucket,
                keys=s3_keys,
                target_dir=cache_directory,
                s3_client=S3_CLIENT,
            )
        except Exception as error:
            raise ModelArtefactError(
                f"Could not obtain model artefacts from S3 with keys {s3_keys}. Cause: {error}"
            ) from error

    @property
    def model_artifact_bucket(self):
        return INFER_BUCKET

    def download_model_artifacts_from_roboflow_api(self) -> None:
        logger.debug("Downloading model artifacts from Roboflow API")
        api_data = get_roboflow_model_data(
            api_key=self.api_key,
            model_id=self.endpoint,
            endpoint_type=ModelEndpointType.ORT,
            device_id=self.device_id,
        )
        if "ort" not in api_data.keys():
            raise ModelArtefactError(
                "Could not find `ort` key in roboflow API model description response."
            )
        api_data = api_data["ort"]
        if "classes" in api_data:
            save_text_lines_in_cache(
                content=api_data["classes"],
                file="class_names.txt",
                model_id=self.endpoint,
            )
        if "model" not in api_data:
            raise ModelArtefactError(
                "Could not find `model` key in roboflow API model description response."
            )
        if "environment" not in api_data:
            raise ModelArtefactError(
                "Could not find `environment` key in roboflow API model description response."
            )
        environment = get_from_url(api_data["environment"])
        model_weights_response = get_from_url(api_data["model"], json_response=False)
        save_bytes_in_cache(
            content=model_weights_response.content,
            file=self.weights_file,
            model_id=self.endpoint,
        )
        if "colors" in api_data:
            environment["COLORS"] = api_data["colors"]
        save_json_in_cache(
            content=environment,
            file="environment.json",
            model_id=self.endpoint,
        )
        if "keypoints_metadata" in api_data:
            # TODO: make sure backend provides that
            save_json_in_cache(
                content=api_data["keypoints_metadata"],
                file="keypoints_metadata.json",
                model_id=self.endpoint,
            )

    def load_model_artifacts_from_cache(self) -> None:
        logger.debug("Model artifacts already downloaded, loading model from cache")
        infer_bucket_files = self.get_all_required_infer_bucket_file()
        if "environment.json" in infer_bucket_files:
            self.environment = load_json_from_cache(
                file="environment.json",
                model_id=self.endpoint,
                object_pairs_hook=OrderedDict,
            )
        if "class_names.txt" in infer_bucket_files:
            self.class_names = load_text_file_from_cache(
                file="class_names.txt",
                model_id=self.endpoint,
                split_lines=True,
                strip_white_chars=True,
            )
        else:
            self.class_names = get_class_names_from_environment_file(
                environment=self.environment
            )
        self.colors = get_color_mapping_from_environment(
            environment=self.environment,
            class_names=self.class_names,
        )
        if "keypoints_metadata.json" in infer_bucket_files:
            self.keypoints_metadata = parse_keypoints_metadata(
                load_json_from_cache(
                    file="keypoints_metadata.json",
                    model_id=self.endpoint,
                    object_pairs_hook=OrderedDict,
                )
            )
        self.num_classes = len(self.class_names)
        if "PREPROCESSING" not in self.environment:
            raise ModelArtefactError(
                "Could not find `PREPROCESSING` key in environment file."
            )
        if issubclass(type(self.environment["PREPROCESSING"]), dict):
            self.preproc = self.environment["PREPROCESSING"]
        else:
            self.preproc = json.loads(self.environment["PREPROCESSING"])
        if self.preproc.get("resize"):
            self.resize_method = self.preproc["resize"].get("format", "Stretch to")
            if self.resize_method not in [
                "Stretch to",
                "Fit (black edges) in",
                "Fit (white edges) in",
            ]:
                self.resize_method = "Stretch to"
        else:
            self.resize_method = "Stretch to"
        logger.debug(f"Resize method is '{self.resize_method}'")
        self.multiclass = self.environment.get("MULTICLASS", False)

    def initialize_model(self) -> None:
        """Initialize the model.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError(self.__class__.__name__ + ".initialize_model")

    def preproc_image(
        self,
        image: Union[Any, InferenceRequestImage],
        disable_preproc_auto_orient: bool = False,
        disable_preproc_contrast: bool = False,
        disable_preproc_grayscale: bool = False,
        disable_preproc_static_crop: bool = False,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Preprocesses an inference request image by loading it, then applying any pre-processing specified by the Roboflow platform, then scaling it to the inference input dimensions.

        Args:
            image (Union[Any, InferenceRequestImage]): An object containing information necessary to load the image for inference.
            disable_preproc_auto_orient (bool, optional): If true, the auto orient preprocessing step is disabled for this call. Default is False.
            disable_preproc_contrast (bool, optional): If true, the contrast preprocessing step is disabled for this call. Default is False.
            disable_preproc_grayscale (bool, optional): If true, the grayscale preprocessing step is disabled for this call. Default is False.
            disable_preproc_static_crop (bool, optional): If true, the static crop preprocessing step is disabled for this call. Default is False.

        Returns:
            Tuple[np.ndarray, Tuple[int, int]]: A tuple containing a numpy array of the preprocessed image pixel data and a tuple of the images original size.
        """
        np_image, is_bgr = load_image(
            image,
            disable_preproc_auto_orient=disable_preproc_auto_orient
            or "auto-orient" not in self.preproc.keys()
            or DISABLE_PREPROC_AUTO_ORIENT,
        )
        preprocessed_image, img_dims = self.preprocess_image(
            np_image,
            disable_preproc_contrast=disable_preproc_contrast,
            disable_preproc_grayscale=disable_preproc_grayscale,
            disable_preproc_static_crop=disable_preproc_static_crop,
        )

        if self.resize_method == "Stretch to":
            resized = cv2.resize(
                preprocessed_image, (self.img_size_w, self.img_size_h), cv2.INTER_CUBIC
            )
        elif self.resize_method == "Fit (black edges) in":
            resized = letterbox_image(
                preprocessed_image, (self.img_size_w, self.img_size_h)
            )
        elif self.resize_method == "Fit (white edges) in":
            resized = letterbox_image(
                preprocessed_image,
                (self.img_size_w, self.img_size_h),
                color=(255, 255, 255),
            )

        if is_bgr:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_in = np.transpose(resized, (2, 0, 1))
        img_in = img_in.astype(np.float32)
        img_in = np.expand_dims(img_in, axis=0)

        return img_in, img_dims

    def preprocess_image(
        self,
        image: np.ndarray,
        disable_preproc_contrast: bool = False,
        disable_preproc_grayscale: bool = False,
        disable_preproc_static_crop: bool = False,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Preprocesses the given image using specified preprocessing steps.

        Args:
            image (Image.Image): The PIL image to preprocess.
            disable_preproc_contrast (bool, optional): If true, the contrast preprocessing step is disabled for this call. Default is False.
            disable_preproc_grayscale (bool, optional): If true, the grayscale preprocessing step is disabled for this call. Default is False.
            disable_preproc_static_crop (bool, optional): If true, the static crop preprocessing step is disabled for this call. Default is False.

        Returns:
            Image.Image: The preprocessed PIL image.
        """
        return prepare(
            image,
            self.preproc,
            disable_preproc_contrast=disable_preproc_contrast,
            disable_preproc_grayscale=disable_preproc_grayscale,
            disable_preproc_static_crop=disable_preproc_static_crop,
        )

    @property
    def weights_file(self) -> str:
        """Abstract property representing the file containing the model weights.

        Raises:
            NotImplementedError: This property must be implemented in subclasses.

        Returns:
            str: The file path to the weights file.
        """
        raise NotImplementedError(self.__class__.__name__ + ".weights_file")


class RoboflowCoreModel(RoboflowInferenceModel):
    """Base Roboflow inference model (Inherits from CvModel since all Roboflow models are CV models currently)."""

    def __init__(
        self,
        model_id: str,
        api_key=None,
    ):
        """Initializes the RoboflowCoreModel instance.

        Args:
            model_id (str): The identifier for the specific model.
            api_key ([type], optional): The API key for authentication. Defaults to None.
        """
        super().__init__(model_id, api_key=api_key)
        self.download_weights()

    def download_weights(self) -> None:
        """Downloads the model weights from the configured source.

        This method includes handling for AWS access keys and error handling.
        """
        infer_bucket_files = self.get_infer_bucket_file_list()
        if are_all_files_cached(files=infer_bucket_files, model_id=self.endpoint):
            logger.debug("Model artifacts already downloaded, loading from cache")
            return None
        if is_model_artefacts_bucket_available():
            self.download_model_artefacts_from_s3()
            return None
        self.download_model_from_roboflow_api()

    def download_model_from_roboflow_api(self) -> None:
        api_data = get_roboflow_model_data(
            api_key=self.api_key,
            model_id=self.endpoint,
            endpoint_type=ModelEndpointType.CORE_MODEL,
            device_id=self.device_id,
        )
        if "weights" not in api_data:
            raise ModelArtefactError(
                f"`weights` key not available in Roboflow API response while downloading model weights."
            )
        for weights_url_key in api_data["weights"]:
            weights_url = api_data["weights"][weights_url_key]
            t1 = perf_counter()
            model_weights_response = get_from_url(weights_url, json_response=False)
            filename = weights_url.split("?")[0].split("/")[-1]
            save_bytes_in_cache(
                content=model_weights_response.content,
                file=filename,
                model_id=self.endpoint,
            )
            if perf_counter() - t1 > 120:
                logger.debug(
                    "Weights download took longer than 120 seconds, refreshing API request"
                )
                api_data = get_roboflow_model_data(
                    api_key=self.api_key,
                    model_id=self.endpoint,
                    endpoint_type=ModelEndpointType.CORE_MODEL,
                    device_id=self.device_id,
                )

    def get_device_id(self) -> str:
        """Returns the device ID associated with this model.

        Returns:
            str: The device ID.
        """
        return self.device_id

    def get_infer_bucket_file_list(self) -> List[str]:
        """Abstract method to get the list of files to be downloaded from the inference bucket.

        Raises:
            NotImplementedError: This method must be implemented in subclasses.

        Returns:
            List[str]: A list of filenames.
        """
        raise NotImplementedError(
            "get_infer_bucket_file_list not implemented for OnnxRoboflowCoreModel"
        )

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Abstract method to preprocess an image.

        Raises:
            NotImplementedError: This method must be implemented in subclasses.

        Returns:
            Image.Image: The preprocessed PIL image.
        """
        raise NotImplementedError(self.__class__.__name__ + ".preprocess_image")

    @property
    def weights_file(self) -> str:
        """Abstract property representing the file containing the model weights. For core models, all model artifacts are handled through get_infer_bucket_file_list method."""
        return None

    @property
    def model_artifact_bucket(self):
        return CORE_MODEL_BUCKET


class OnnxRoboflowInferenceModel(RoboflowInferenceModel):
    """Roboflow Inference Model that operates using an ONNX model file."""

    def __init__(
        self,
        model_id: str,
        onnxruntime_execution_providers: List[
            str
        ] = get_onnxruntime_execution_providers(ONNXRUNTIME_EXECUTION_PROVIDERS),
        *args,
        **kwargs,
    ):
        """Initializes the OnnxRoboflowInferenceModel instance.

        Args:
            model_id (str): The identifier for the specific ONNX model.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(model_id, *args, **kwargs)
        if self.load_weights or not self.has_model_metadata:
            self.onnxruntime_execution_providers = onnxruntime_execution_providers
            expanded_execution_providers = []
            for ep in self.onnxruntime_execution_providers:
                if ep == "TensorrtExecutionProvider":
                    ep = (
                        "TensorrtExecutionProvider",
                        {
                            "trt_engine_cache_enable": True,
                            "trt_engine_cache_path": os.path.join(
                                TENSORRT_CACHE_PATH, self.endpoint
                            ),
                            "trt_fp16_enable": True,
                        },
                    )
                expanded_execution_providers.append(ep)
            self.onnxruntime_execution_providers = expanded_execution_providers

        self.initialize_model()
        self.image_loader_threadpool = ThreadPoolExecutor(max_workers=None)
        try:
            self.validate_model()
        except ModelArtefactError as e:
            logger.error(f"Unable to validate model artifacts, clearing cache: {e}")
            self.clear_cache()
            raise ModelArtefactError from e

    def infer(self, image: Any, **kwargs) -> Any:
        """Runs inference on given data.
        - image:
            can be a BGR numpy array, filepath, InferenceRequestImage, PIL Image, byte-string, etc.
        """
        input_elements = len(image) if isinstance(image, list) else 1
        max_batch_size = MAX_BATCH_SIZE if self.batching_enabled else self.batch_size
        if (input_elements == 1) or (max_batch_size == float("inf")):
            return super().infer(image, **kwargs)
        logger.debug(
            f"Inference will be executed in batches, as there is {input_elements} input elements and "
            f"maximum batch size for a model is set to: {max_batch_size}"
        )
        inference_results = []
        for batch_input in create_batches(sequence=image, batch_size=max_batch_size):
            batch_inference_results = super().infer(batch_input, **kwargs)
            inference_results.append(batch_inference_results)
        return self.merge_inference_results(inference_results=inference_results)

    def merge_inference_results(self, inference_results: List[Any]) -> Any:
        return list(itertools.chain(*inference_results))

    def validate_model(self) -> None:
        if MODEL_VALIDATION_DISABLED:
            logger.debug("Model validation disabled.")
            return None
        logger.debug("Starting model validation")
        if not self.load_weights:
            return
        try:
            assert self.onnx_session is not None
        except AssertionError as e:
            raise ModelArtefactError(
                "ONNX session not initialized. Check that the model weights are available."
            ) from e
        try:
            self.run_test_inference()
        except Exception as e:
            raise ModelArtefactError(f"Unable to run test inference. Cause: {e}") from e
        try:
            self.validate_model_classes()
        except Exception as e:
            raise ModelArtefactError(
                f"Unable to validate model classes. Cause: {e}"
            ) from e
        logger.debug("Model validation finished")

    def run_test_inference(self) -> None:
        test_image = (np.random.rand(1024, 1024, 3) * 255).astype(np.uint8)
        logger.debug(f"Running test inference. Image size: {test_image.shape}")
        result = self.infer(test_image)
        logger.debug(f"Test inference finished.")
        return result

    def get_model_output_shape(self) -> Tuple[int, int, int]:
        test_image = (np.random.rand(1024, 1024, 3) * 255).astype(np.uint8)
        logger.debug(f"Getting model output shape. Image size: {test_image.shape}")
        test_image, _ = self.preprocess(test_image)
        output = self.predict(test_image)[0]
        logger.debug(f"Model output shape test finished.")
        return output.shape

    def validate_model_classes(self) -> None:
        pass

    def get_infer_bucket_file_list(self) -> list:
        """Returns the list of files to be downloaded from the inference bucket for ONNX model.

        Returns:
            list: A list of filenames specific to ONNX models.
        """
        return ["environment.json", "class_names.txt"]

    def initialize_model(self) -> None:
        """Initializes the ONNX model, setting up the inference session and other necessary properties."""
        logger.debug("Getting model artefacts")
        self.get_model_artifacts()
        logger.debug("Creating inference session")
        if self.load_weights or not self.has_model_metadata:
            t1_session = perf_counter()
            # Create an ONNX Runtime Session with a list of execution providers in priority order. ORT attempts to load providers until one is successful. This keeps the code across devices identical.
            providers = self.onnxruntime_execution_providers

            if not self.load_weights:
                providers = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
            try:
                session_options = onnxruntime.SessionOptions()
                # TensorRT does better graph optimization for its EP than onnx
                if has_trt(providers):
                    session_options.graph_optimization_level = (
                        onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
                    )
                self.onnx_session = onnxruntime.InferenceSession(
                    self.cache_file(self.weights_file),
                    providers=providers,
                    sess_options=session_options,
                )
            except Exception as e:
                self.clear_cache()
                raise ModelArtefactError(
                    f"Unable to load ONNX session. Cause: {e}"
                ) from e
            logger.debug(f"Session created in {perf_counter() - t1_session} seconds")

            if REQUIRED_ONNX_PROVIDERS:
                available_providers = onnxruntime.get_available_providers()
                for provider in REQUIRED_ONNX_PROVIDERS:
                    if provider not in available_providers:
                        raise OnnxProviderNotAvailable(
                            f"Required ONNX Execution Provider {provider} is not availble. Check that you are using the correct docker image on a supported device."
                        )

            inputs = self.onnx_session.get_inputs()[0]
            input_shape = inputs.shape
            self.batch_size = input_shape[0]
            self.img_size_h = input_shape[2]
            self.img_size_w = input_shape[3]
            self.input_name = inputs.name
            if isinstance(self.img_size_h, str) or isinstance(self.img_size_w, str):
                if "resize" in self.preproc:
                    self.img_size_h = int(self.preproc["resize"]["height"])
                    self.img_size_w = int(self.preproc["resize"]["width"])
                else:
                    self.img_size_h = 640
                    self.img_size_w = 640

            if isinstance(self.batch_size, str):
                self.batching_enabled = True
                logger.debug(
                    f"Model {self.endpoint} is loaded with dynamic batching enabled"
                )
            else:
                self.batching_enabled = False
                logger.debug(
                    f"Model {self.endpoint} is loaded with dynamic batching disabled"
                )

            model_metadata = {
                "batch_size": self.batch_size,
                "img_size_h": self.img_size_h,
                "img_size_w": self.img_size_w,
            }
            logger.debug(f"Writing model metadata to memcache")
            self.write_model_metadata_to_memcache(model_metadata)
            if not self.load_weights:  # had to load weights to get metadata
                del self.onnx_session
        else:
            if not self.has_model_metadata:
                raise ValueError(
                    "This should be unreachable, should get weights if we don't have model metadata"
                )
            logger.debug(f"Loading model metadata from memcache")
            metadata = self.model_metadata_from_memcache()
            self.batch_size = metadata["batch_size"]
            self.img_size_h = metadata["img_size_h"]
            self.img_size_w = metadata["img_size_w"]
            if isinstance(self.batch_size, str):
                self.batching_enabled = True
                logger.debug(
                    f"Model {self.endpoint} is loaded with dynamic batching enabled"
                )
            else:
                self.batching_enabled = False
                logger.debug(
                    f"Model {self.endpoint} is loaded with dynamic batching disabled"
                )
        logger.debug("Model initialisation finished.")

    def load_image(
        self,
        image: Any,
        disable_preproc_auto_orient: bool = False,
        disable_preproc_contrast: bool = False,
        disable_preproc_grayscale: bool = False,
        disable_preproc_static_crop: bool = False,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        if isinstance(image, list):
            preproc_image = partial(
                self.preproc_image,
                disable_preproc_auto_orient=disable_preproc_auto_orient,
                disable_preproc_contrast=disable_preproc_contrast,
                disable_preproc_grayscale=disable_preproc_grayscale,
                disable_preproc_static_crop=disable_preproc_static_crop,
            )
            imgs_with_dims = self.image_loader_threadpool.map(preproc_image, image)
            imgs, img_dims = zip(*imgs_with_dims)
            img_in = np.concatenate(imgs, axis=0)
        else:
            img_in, img_dims = self.preproc_image(
                image,
                disable_preproc_auto_orient=disable_preproc_auto_orient,
                disable_preproc_contrast=disable_preproc_contrast,
                disable_preproc_grayscale=disable_preproc_grayscale,
                disable_preproc_static_crop=disable_preproc_static_crop,
            )
            img_dims = [img_dims]
        return img_in, img_dims

    @property
    def weights_file(self) -> str:
        """Returns the file containing the ONNX model weights.

        Returns:
            str: The file path to the weights file.
        """
        return "weights.onnx"


class OnnxRoboflowCoreModel(RoboflowCoreModel):
    """Roboflow Inference Model that operates using an ONNX model file."""

    pass


def get_class_names_from_environment_file(environment: Optional[dict]) -> List[str]:
    if environment is None:
        raise ModelArtefactError(
            f"Missing environment while attempting to get model class names."
        )
    if class_mapping_not_available_in_environment(environment=environment):
        raise ModelArtefactError(
            f"Missing `CLASS_MAP` in environment or `CLASS_MAP` is not dict."
        )
    class_names = []
    for i in range(len(environment["CLASS_MAP"].keys())):
        class_names.append(environment["CLASS_MAP"][str(i)])
    return class_names


def class_mapping_not_available_in_environment(environment: dict) -> bool:
    return "CLASS_MAP" not in environment or not issubclass(
        type(environment["CLASS_MAP"]), dict
    )


def get_color_mapping_from_environment(
    environment: Optional[dict], class_names: List[str]
) -> Dict[str, str]:
    if color_mapping_available_in_environment(environment=environment):
        return environment["COLORS"]
    return {
        class_name: DEFAULT_COLOR_PALETTE[i % len(DEFAULT_COLOR_PALETTE)]
        for i, class_name in enumerate(class_names)
    }


def color_mapping_available_in_environment(environment: Optional[dict]) -> bool:
    return (
        environment is not None
        and "COLORS" in environment
        and issubclass(type(environment["COLORS"]), dict)
    )


def is_model_artefacts_bucket_available() -> bool:
    return (
        AWS_ACCESS_KEY_ID is not None
        and AWS_SECRET_ACCESS_KEY is not None
        and LAMBDA
        and S3_CLIENT is not None
    )


def parse_keypoints_metadata(metadata: list) -> dict:
    return {
        e["object_class_id"]: {int(key): value for key, value in e["keypoints"].items()}
        for e in metadata
    }



from abc import abstractmethod
from time import perf_counter
from typing import Any, List, Tuple, Union

import numpy as np

from inference.core.cache.model_artifacts import clear_cache, initialise_cache
from inference.core.entities.requests.inference import InferenceRequest
from inference.core.entities.responses.inference import InferenceResponse, StubResponse
from inference.core.models.base import Model
from inference.core.models.types import PreprocessReturnMetadata
from inference.core.utils.image_utils import np_image_to_base64


class ModelStub(Model):
    def __init__(self, model_id: str, api_key: str):
        super().__init__()
        self.model_id = model_id
        self.api_key = api_key
        self.dataset_id, self.version_id = model_id.split("/")
        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}
        initialise_cache(model_id=model_id)

    def infer_from_request(
        self, request: InferenceRequest
    ) -> Union[InferenceResponse, List[InferenceResponse]]:
        t1 = perf_counter()
        stub_prediction = self.infer(**request.dict())
        response = self.make_response(request=request, prediction=stub_prediction)
        response.time = perf_counter() - t1
        return response

    def infer(self, *args, **kwargs) -> Any:
        _ = self.preprocess()
        dummy_prediction = self.predict()
        return self.postprocess(dummy_prediction)

    def preprocess(
        self, *args, **kwargs
    ) -> Tuple[np.ndarray, PreprocessReturnMetadata]:
        return np.zeros((128, 128, 3), dtype=np.uint8), {}  # type: ignore

    def predict(self, *args, **kwargs) -> Tuple[np.ndarray, ...]:
        return (np.zeros((1, 8)),)

    def postprocess(self, predictions: Tuple[np.ndarray, ...], *args, **kwargs) -> Any:
        return {
            "is_stub": True,
            "model_id": self.model_id,
        }

    def clear_cache(self) -> None:
        clear_cache(model_id=self.model_id)

    @abstractmethod
    def make_response(
        self, request: InferenceRequest, prediction: dict, **kwargs
    ) -> Union[InferenceResponse, List[InferenceResponse]]:
        pass


class ClassificationModelStub(ModelStub):
    task_type = "classification"

    def make_response(
        self, request: InferenceRequest, prediction: dict, **kwargs
    ) -> Union[InferenceResponse, List[InferenceResponse]]:
        stub_visualisation = None
        if getattr(request, "visualize_predictions", False):
            stub_visualisation = np_image_to_base64(
                np.zeros((128, 128, 3), dtype=np.uint8)
            )
        return StubResponse(
            is_stub=prediction["is_stub"],
            model_id=prediction["model_id"],
            task_type=self.task_type,
            visualization=stub_visualisation,
        )


class ObjectDetectionModelStub(ModelStub):
    task_type = "object-detection"

    def make_response(
        self, request: InferenceRequest, prediction: dict, **kwargs
    ) -> Union[InferenceResponse, List[InferenceResponse]]:
        stub_visualisation = None
        if getattr(request, "visualize_predictions", False):
            stub_visualisation = np_image_to_base64(
                np.zeros((128, 128, 3), dtype=np.uint8)
            )
        return StubResponse(
            is_stub=prediction["is_stub"],
            model_id=prediction["model_id"],
            task_type=self.task_type,
            visualization=stub_visualisation,
        )


class InstanceSegmentationModelStub(ModelStub):
    task_type = "instance-segmentation"

    def make_response(
        self, request: InferenceRequest, prediction: dict, **kwargs
    ) -> Union[InferenceResponse, List[InferenceResponse]]:
        stub_visualisation = None
        if getattr(request, "visualize_predictions", False):
            stub_visualisation = np_image_to_base64(
                np.zeros((128, 128, 3), dtype=np.uint8)
            )
        return StubResponse(
            is_stub=prediction["is_stub"],
            model_id=prediction["model_id"],
            task_type=self.task_type,
            visualization=stub_visualisation,
        )


class KeypointsDetectionModelStub(ModelStub):
    task_type = "keypoint-detection"

    def make_response(
        self, request: InferenceRequest, prediction: dict, **kwargs
    ) -> Union[InferenceResponse, List[InferenceResponse]]:
        stub_visualisation = None
        if getattr(request, "visualize_predictions", False):
            stub_visualisation = np_image_to_base64(
                np.zeros((128, 128, 3), dtype=np.uint8)
            )
        return StubResponse(
            is_stub=prediction["is_stub"],
            model_id=prediction["model_id"],
            task_type=self.task_type,
            visualization=stub_visualisation,
        )



from typing import Dict, NewType

PreprocessReturnMetadata = NewType("PreprocessReturnMetadata", Dict)


def get_num_classes_from_model_prediction_shape(len_prediction, masks=0, keypoints=0):
    num_classes = len_prediction - 5 - masks - (keypoints * 3)
    return num_classes



from typing import Dict, List, Tuple, Union


def has_trt(providers: List[Union[Tuple[str, Dict], str]]) -> bool:
    for p in providers:
        if isinstance(p, tuple):
            name = p[0]
        else:
            name = p
        if name == "TensorrtExecutionProvider":
            return True
    return False



from typing import List

from inference.core.entities.responses.inference import Keypoint
from inference.core.exceptions import ModelArtefactError


def superset_keypoints_count(keypoints_metadata={}) -> int:
    """Returns the number of keypoints in the superset."""
    max_keypoints = 0
    for keypoints in keypoints_metadata.values():
        if len(keypoints) > max_keypoints:
            max_keypoints = len(keypoints)
    return max_keypoints


def model_keypoints_to_response(
    keypoints_metadata: dict,
    keypoints: List[float],
    predicted_object_class_id: int,
    keypoint_confidence_threshold: float,
) -> List[Keypoint]:
    if keypoints_metadata is None:
        raise ModelArtefactError("Keypoints metadata not available.")
    keypoint_id2name = keypoints_metadata[predicted_object_class_id]
    results = []
    for keypoint_id in range(len(keypoints) // 3):
        if keypoint_id >= len(keypoint_id2name):
            # Ultralytics only supports single class keypoint detection, so points might be padded with zeros
            break
        confidence = keypoints[3 * keypoint_id + 2]
        if confidence < keypoint_confidence_threshold:
            continue
        keypoint = Keypoint(
            x=keypoints[3 * keypoint_id],
            y=keypoints[3 * keypoint_id + 1],
            confidence=confidence,
            class_id=keypoint_id,
            class_name=keypoint_id2name[keypoint_id],
        )
        results.append(keypoint)
    return results



