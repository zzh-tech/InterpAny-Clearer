import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.onnx import SamOnnxModel
# import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic
import warnings

class Embedding():
    def __init__(self, outDir='./data/results', modelDir='./data/models'):
        ## Set up internal variable
        # pprint(self.args['test'])
        self.outDir = outDir

        ## Check needed files
        # print(dataset_path)
        self.checkpoint = f'{modelDir}/sam_vit_b_01ec64.pth'
        print(self.checkpoint)
        if not os.path.exists(self.checkpoint):
            raise FileExistsError('"sam_vit_b_01ec64.pth" does not exists.')
        
        ## variables
        self.model_type = "vit_b"
        self.sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)

        ### start code to generate onnx model
        self.onnx_model_quantized_path = f'{modelDir}/sam_onnx_quantized_example.onnx'
        self.onnx_model_path = f'{modelDir}/sam_onnx_example.onnx'
        hasModel = os.path.isfile(f'{modelDir}/sam_onnx_quantized_example.onnx')
        if not hasModel:
            onnx_model = SamOnnxModel(self.sam, return_single_mask=True)

            dynamic_axes = {
                "point_coords": {1: "num_points"},
                "point_labels": {1: "num_points"},
            }

            embed_dim = self.sam.prompt_encoder.embed_dim
            embed_size = self.sam.prompt_encoder.image_embedding_size
            mask_input_size = [4 * x for x in embed_size]
            dummy_inputs = {
                "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
                "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
                "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
                "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
                "has_mask_input": torch.tensor([1], dtype=torch.float),
                "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
            }
            output_names = ["masks", "iou_predictions", "low_res_masks"]

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                with open(self.onnx_model_path, "wb") as f:
                    torch.onnx.export(
                        onnx_model,
                        tuple(dummy_inputs.values()),
                        f,
                        export_params=True,
                        verbose=False,
                        opset_version=13,
                        do_constant_folding=True,
                        input_names=list(dummy_inputs.keys()),
                        output_names=output_names,
                        dynamic_axes=dynamic_axes,
                    )    


            self.onnx_model_quantized_path = f'{modelDir}/sam_onnx_quantized_example.onnx'
            quantize_dynamic(
                model_input=self.onnx_model_path,
                model_output=self.onnx_model_quantized_path,
                optimize_model=True,
                per_channel=False,
                reduce_range=False,
                weight_type=QuantType.QUInt8,
            )
        ### end code to generate onnx model

    def show_mask(mask, ax):
        color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
        
    def show_points(coords, labels, ax, marker_size=375):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
        
    def show_box(box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))


    def generateEmbedding(self, filename):
        image = cv2.imread(f'data/uploads/{filename}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ### Create embeding
        # ort_session = onnxruntime.InferenceSession(self.onnx_model_quantized_path)

        # SAM
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # print(device)
        self.sam.to(device=device)
        predictor = SamPredictor(self.sam)
        predictor.set_image(image)
        image_embedding = predictor.get_image_embedding().cpu().numpy()
        print(image_embedding.shape)

        ### Save embedding to file
        embeddingFile = os.path.splitext(filename)[0]
        embeddingFileUrl = f'data/embeddings/{embeddingFile}.npy'
        np.save(embeddingFileUrl, image_embedding)
        return f'http://localhost:5001/embeddings/{embeddingFile}.npy'
        
### End: Embedding

if __name__ == '__main__':
    print("test script")
    # Example
    dcm = Embedding()
    dcm.generateEmbedding()
    print("end test script")

