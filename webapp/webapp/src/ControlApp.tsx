import React, { useContext, useState } from 'react';
import { Button, Input } from '@material-tailwind/react';
import MaskList from './components/MaskList';
import { ControlProps, modelControlProps, modelMaskProps, modelRawMaskProps } from './components/helpers/Interfaces';
import { ControlContext } from './components/hooks/createContext';
import { onnxMaskToImage } from './components/helpers/maskUtils';
const _ = require('lodash');
const axios = require("axios");
const SERVER_URL = "http://127.0.0.1:5001";

const ControlApp = ({defaultRawMask, masks, handleDelete, image1Path, image2Path, updateMasks, updateLoader} : ControlProps) => {
  const {
    controls: [controls, setControls],
    selectedIndices: [selectedIndices, setSelectedIndices]
  } = useContext(ControlContext)!;

  const handleCombineDelete = (index: number) => {
    handleDelete(index);
    const newControls = [...controls];
    newControls.splice(index,1)
    setControls(newControls);
  };

  const [videoUrl, setVideoUrl] = useState('');
  const [samplingPoints, setSamplingPoints] = useState('');

  const combineMasks = () => {
    console.log("vatran combineMasks", selectedIndices, masks.length)

    let unselectedMasks = [];

    const first = selectedIndices[0]
    
    // for control update
    const newControls = [...controls].filter(function(v, i) {
      return selectedIndices.indexOf(i) < 0
    });
    

    const combinedControl = controls[first];


    const resultMask: modelMaskProps = {
      id: masks[first].id,
      name: masks[first].name,
      maskImg: masks[first].maskImg,
      rawMask: {
        width: masks[first].rawMask.width,
        height: masks[first].rawMask.height,
        data: masks[first].rawMask.data,
      }
    }
    const length = resultMask.rawMask.data.length;
    for (let i = 1; i < selectedIndices.length; i++) {
      const data = masks[selectedIndices[i]].rawMask.data;

      for (let j = 0; j < length; j++) {
        if (data[j] > 0) {
          resultMask.rawMask.data[j] = 1;
        }
      }
    }
    newControls.unshift(combinedControl);
    setControls(newControls); // update controls after combining

    // update maskImg of resultMask
    resultMask.maskImg = onnxMaskToImage(
      resultMask.rawMask.data, 
      resultMask.rawMask.height, 
      resultMask.rawMask.width
    );

    // the combined
    unselectedMasks.push(resultMask);

    // plus the unselected
    for (let i = 0; i < masks.length; i++) {
      if (selectedIndices.indexOf(i) < 0) {
        unselectedMasks.push(masks[i])
      }
    }
    
    updateMasks(unselectedMasks)
    // reset selected indices after merge
    setSelectedIndices([])
  }

  /*
  const interpolateImages = () => {
    // show loading
    updateLoader(true);
    console.log("vatran filenames", image1Path, image2Path)

    const json = {
      filename1: image1Path,
      filename2: image2Path
    }
    axios.post(`${SERVER_URL}/interpolate`, json, {
      headers: {
        "Content-Type": "application/json",
      },
    })
    .then((response: any) => {
      console.log("vatran response", response);
      setVideoUrl(response['data']['video_url'])
      updateLoader(false);
    }, (error: any) => {
      console.log(error);
      updateLoader(false);
    });
  }
  */

  const clearMasksAndVideo = () => {
    setControls([]);
    updateMasks([]);
    setVideoUrl('');
  }

  const exportMasksAndInterpolate = () => {
    setVideoUrl('');
    exportCombinedMasksAndInterpolate(masks, controls, defaultRawMask);
  }

  const exportCombinedMasksAndInterpolate = (maskArray: Array<modelMaskProps>, controlArray: Array<modelControlProps>, defaultRawMask: modelRawMaskProps | null) => {
    updateLoader(true);
    
    const outputControls = [];
    const outputMasks = [];

    if (maskArray.length > 0) {
      for (let i = 0; i < maskArray.length; i++) {
        const control = controlArray[i];
        const mask = maskArray[i];

        outputControls.push(control.points);
        const arr = reshape(mask.rawMask?.data!, mask.rawMask?.width!)
        outputMasks.push(arr);
      }
    } else {
      outputControls.push([0,20,40,60,80,100]);
      const arr = reshape(defaultRawMask?.data!, defaultRawMask?.width!)
      outputMasks.push(arr);
    }
    
    const fileData = JSON.stringify(
      {
        masks: outputMasks, 
        controls: outputControls, 
        image1_url: image1Path,
        image2_url: image2Path,
        sampling_points: samplingPoints
      }
    );

    /* old code to download mask_dict
    const blob = new Blob([fileData], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.download = `masks_dict.txt`;
    link.href = url;
    link.click(); 
    */

    const blob = new Blob([fileData], { type: "text/plain" });
    const url = URL.createObjectURL(blob);

    // create form data to upload mask_dict
    let formData = new FormData();
    formData.append("file", blob);
  
    axios.post(`${SERVER_URL}/upload_mask_dict`, formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    })
    .then((response: any) => {
      updateLoader(false);
      setVideoUrl(response['data']['video_url'])
    }, (error: any) => {
      console.log(error);
      updateLoader(false);
    });
  }

  const reshape = (array: Uint8ClampedArray, width: number) => {
    const inputArr = Array.from(array)
    const newArr = [];
    while(inputArr.length) newArr.push(inputArr.splice(0,width));
    return newArr;
  }

  const handleSamplingChange = (event: React.FormEvent<HTMLInputElement>) => {
    setSamplingPoints(event.currentTarget.value);
  };
  
  return (
    <div className="column h-full">
      <div className="row scrollbable">
        <Button className="m-2" size="md" onClick={combineMasks} disabled={selectedIndices.length <= 1} onResize={undefined} onResizeCapture={undefined}>Combine Masks</Button>
        <Button className="m-2" size="md" onClick={exportMasksAndInterpolate} disabled={(masks.length == 0 && defaultRawMask == null) || samplingPoints.length == 0 || image2Path == ""} onResize={undefined} onResizeCapture={undefined}>Interpolate</Button>
        <Button className="m-2" size="md" onClick={clearMasksAndVideo} onResize={undefined} onResizeCapture={undefined}>Clear</Button>
        
        <div className="w-12 m-2">
          <Input size="md" label="Sampling points" type="number" maxLength={4} onChange={handleSamplingChange} value={samplingPoints} onResize={undefined} onResizeCapture={undefined} crossOrigin={undefined}/>
        </div>
        {videoUrl && videoUrl.length > 0 &&
          <div className="m-2">
            <video controls width="100%"><source src={videoUrl} type="video/mp4"/></video>
          </div>
        }
        <MaskList masks={masks} handleDelete={handleCombineDelete}/>
      </div>
    </div>
  );
}

export default ControlApp;