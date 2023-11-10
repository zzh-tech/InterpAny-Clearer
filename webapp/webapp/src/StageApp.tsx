// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.

// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import { InferenceSession, Tensor } from "onnxruntime-web";
import React, { useContext, useEffect, useState } from "react";
import "./assets/scss/App.scss";
import { handleImageScale } from "./components/helpers/scaleHelper";
import { modelScaleProps, StageAppProps } from "./components/helpers/Interfaces";
import { onnxMaskToImage, arrayToMaskArray, allOnesMaskArray } from "./components/helpers/maskUtils";
import { modelData } from "./components/helpers/onnxModelAPI";
import DragDropFile from "./components/DragDropFile";
import Stage from "./components/Stage";
import { AppContext } from "./components/hooks/createContext";
const ort = require("onnxruntime-web");
const axios = require("axios");
const SERVER_URL = "http://127.0.0.1:5001";

/* @ts-ignore */
import npyjs from "npyjs";

// Define image, embedding and model paths
const MODEL_DIR = "/model/sam_onnx_quantized_example_fast.onnx";

const StageApp = ({ setDefaultRawMask, addMask, updateLoader, onUploadImage1, onUploadImage2 }: StageAppProps) => {
  const {
    clicks: [clicks],
    image: [image, setImage],
    rawMask: [, setRawMask],
    maskImg: [, setMaskImg],
  } = useContext(AppContext)!;
  const [model, setModel] = useState<InferenceSession | null>(null); // ONNX model
  const [tensor, setTensor] = useState<Tensor | null>(null); // Image embedding tensor

  // The ONNX model expects the input to be rescaled to 1024. 
  // The modelScale state variable keeps track of the scale values.
  const [modelScale, setModelScale] = useState<modelScaleProps | null>(null);

  const [image2, setImage2] = useState<HTMLImageElement | null>(null);

  const [currentFile, setCurrentFile] = useState<File | null>();
  const [currentFile2, setCurrentFile2] = useState<File | null>();

  // Initialize the ONNX model. load the image, and load the SAM
  // pre-computed image embedding
  useEffect(() => {
    // Initialize the ONNX model
    const initModel = async () => {
      try {
        if (MODEL_DIR === undefined) return;
        const URL: string = MODEL_DIR;
        const model = await InferenceSession.create(URL);
        setModel(model);
      } catch (e) {
        console.log(e);
      }
    };
    initModel();

  }, []);

  // START image2 resize code
  // Determine if we should shrink or grow the images to match the
  // width or the height of the page and setup a ResizeObserver to
  // monitor changes in the size of the page
  const [shouldFitToWidth, setShouldFitToWidth] = useState(true);
  const bodyEl = document.body;
  const fitToPage = () => {
    if (!image2) return;
    const imageAspectRatio = image2.width / image2.height;
    const screenAspectRatio = window.innerWidth / window.innerHeight;
    setShouldFitToWidth(imageAspectRatio > screenAspectRatio);
  };
  const resizeObserver = new ResizeObserver((entries) => {
    for (const entry of entries) {
      if (entry.target === bodyEl) {
        fitToPage();
      }
    }
  });
  useEffect(() => {
    fitToPage();
    resizeObserver.observe(bodyEl);
    return () => {
      resizeObserver.unobserve(bodyEl);
    };
  }, [image2]);

  const imageClasses = "";
  // END image2 resize code

  const loadImage = async (url: URL) => {
    try {
      const img = new Image();
      img.src = url.href;
      img.onload = () => {
        const { height, width, samScale } = handleImageScale(img);
        setModelScale({
          height: height,  // original image height
          width: width,  // original image width
          samScale: samScale, // scaling factor for image which has been resized to longest side 1024
        });
        img.width = width; 
        img.height = height; 

        setDefaultRawMask({width:width, height:height, data:allOnesMaskArray(width, height)})
        setImage(img);
      };
    } catch (error) {
      console.log(error);
    }
  };

  const loadImage2 = async (url: URL) => {
    try {
      const img = new Image();
      img.src = url.href;
      img.onload = () => {
        const { height, width } = handleImageScale(img);
        img.width = width; 
        img.height = height; 
        setImage2(img);
      };
    } catch (error) {
      console.log(error);
    }
  };

  // Decode a Numpy file into a tensor. 
  const loadNpyTensor = async (tensorFile: string, dType: string) => {
    let npLoader = new npyjs();
    const npArray = await npLoader.load(tensorFile);
    const tensor = new ort.Tensor(dType, npArray.data, npArray.shape);
    return tensor;
  };

  // Run the ONNX model every time clicks has changed
  useEffect(() => {
    runONNX();
  }, [clicks]);

  const runONNX = async () => {
    try {
      if (
        model === null ||
        clicks === null ||
        tensor === null ||
        modelScale === null
      )
        return;
      else {
        // Preapre the model input in the correct format for SAM. 
        // The modelData function is from onnxModelAPI.tsx.
        const feeds = modelData({
          clicks,
          tensor,
          modelScale,
        });
        if (feeds === undefined) return;
        // Run the SAM ONNX model with the feeds returned from modelData()
        const results = await model.run(feeds);
        const output = results[model.outputNames[0]];
        // The predicted mask returned from the ONNX model is an array which is 
        // rendered as an HTML image using onnxMaskToImage() from maskUtils.tsx.
        // console.log("vatran width", output.dims[3])
        // console.log("vatran height", output.dims[2])
        setRawMask({width:output.dims[3], height:output.dims[2], data:arrayToMaskArray(output.data)})
        setMaskImg(onnxMaskToImage(output.data, output.dims[2], output.dims[3]));
      }
    } catch (e) {
      console.log(e);
    }
  };

  // const onFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
  //   const { files } = event.target;
  //   const selectedFiles = files as FileList;
  //   setCurrentFile(selectedFiles?.[0]);
  // };

  // const uploadCurrentFile = () => {
  //   let formData = new FormData();
    
  //   formData.append("file", currentFile!);
  
  //   axios.post(`${SERVER_URL}/upload`, formData, {
  //     headers: {
  //       "Content-Type": "multipart/form-data",
  //     },
  //   })
  //   .then((response: any) => {
  //     console.log(response);
  //     const url = new URL(response['data']['url'], 'http://localhost:5001');
  //     const filename = response['data']['filename'];
  //     loadImage(url);
  //     onUploadImage1(url.href);
  //     // show loader, and create embedding
  //     getEmbedding(filename);
  //   }, (error: any) => {
  //     console.log(error);
  //   });
  // };
  
  const getEmbedding = (filename: string) => {
    // show loading
    updateLoader(true);
    console.log("vatran filename", filename)

    const json = {
      filename
    }
    axios.post(`${SERVER_URL}/embedding`, json, {
      headers: {
        "Content-Type": "application/json",
      },
    })
    .then((response: any) => {
      console.log("vatran response", response);
      // Load the Segment Anything pre-computed embedding
      const filenameEmbedding = response['data']['embedding_url'];
      Promise.resolve(loadNpyTensor(filenameEmbedding, "float32")).then(
        (embedding) => setTensor(embedding)
      );
      updateLoader(false);
    }, (error: any) => {
      console.log(error);
      updateLoader(false);
    });
  }

  // file 1
  const onImage1Uploaded = (url: URL, fileName: string) => {
    loadImage(url);
    onUploadImage1(url.href);
    // show loader, and create embedding
    getEmbedding(fileName);
  }

  // file 2
  const onImage2Uploaded = (url: URL, fileName: string) => {
    loadImage2(url);
    onUploadImage2(url.href);
  }

  const clearImage2 = (event: { target: any; }) => {
    setImage2(null);
    onUploadImage2("");
  };

  // view returned
  const flexCenterClasses = "flex items-center justify-center";
  return (
    <div className="column h-full">
      <div className="row h-[50%] mb-16">
        {
          !image && (
            <div className={`${flexCenterClasses} w-full h-full`}>
              <div className={`${flexCenterClasses} relative w-[90%] h-[85%]`}>
                <DragDropFile message="Drag and drop image file for start frame or" uploadedCallback={onImage1Uploaded}/>
              </div>
            </div>
          )
        }

        <Stage setDefaultRawMask={setDefaultRawMask} addMask={addMask}/>
        
      </div>
      <div className="row h-[50%]">
        <div className={`${flexCenterClasses} w-full h-full`}>
          <div className={`${flexCenterClasses} relative w-[90%] h-[85%]`}>
            {
              !image2 && (
                <DragDropFile message="Drag and drop image file for end frame or" uploadedCallback={onImage2Uploaded}/>
              )
            }
            {image2 && (
              <div className={`${flexCenterClasses} relative w-full h-full image-container`}>
                <img src='/images/close_button.png' className='close-button' onClick={clearImage2}/>
                <img
                  src={image2.src}
                  className={`${
                    shouldFitToWidth ? "w-full" : "h-full"
                  } ${imageClasses}`}
                ></img>
                
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default StageApp;
