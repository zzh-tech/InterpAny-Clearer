// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.

// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import React, { useContext } from "react";
import * as _ from "underscore";
import Tool from "./Tool";
import { StageProps, modelInputProps } from "./helpers/Interfaces";
import { AppContext } from "./hooks/createContext";
import { v4 as uuidv4 } from 'uuid';

const Stage = ({ setDefaultRawMask, addMask }: StageProps) => {
  const {
    clicks: [, setClicks],
    image: [image, setImage],
    rawMask: [rawMask],
    maskImg: [maskImg, setMaskImg],
  } = useContext(AppContext)!;

  const getClick = (x: number, y: number): modelInputProps => {
    const clickType = 1;
    return { x, y, clickType };
  };

  // Get mouse position and scale the (x, y) coordinates back to the natural
  // scale of the image. Update the state of clicks with setClicks to trigger
  // the ONNX model to run and generate a new mask via a useEffect in App.tsx
  const handleMouseMove = _.throttle((e: any) => {
    let el = e.nativeEvent.target;
    const rect = el.getBoundingClientRect();
    let x = e.clientX - rect.left;
    let y = e.clientY - rect.top;
    const imageScale = image ? image.width / el.offsetWidth : 1;
    x *= imageScale;
    y *= imageScale;
    const click = getClick(x, y);
    if (click) setClicks([click]);
  }, 15);

  const handleMouseClick = (e: any) => {
    let el = e.nativeEvent.target;
    const rect = el.getBoundingClientRect();
    let x = e.clientX - rect.left;
    let y = e.clientY - rect.top;
    const imageScale = image ? image.width / el.offsetWidth : 1;
    x *= imageScale;
    y *= imageScale;

    // add a mask to the mask panel
    if (maskImg && rawMask) {
      addMask(
        {
          id: uuidv4(), 
          name: `${length}`, 
          maskImg: maskImg,
          rawMask: rawMask
        }
      );
    }
  }

  const clearImage1 = (event: { target: any; }) => {
    setImage(null);
    setDefaultRawMask(null);
    setMaskImg(null);
  };

  const flexCenterClasses = "flex items-center justify-center";
  return (
    <div className={`${flexCenterClasses} w-full h-full`}>
      <div className={`${flexCenterClasses} relative w-[90%] h-[85%]`}>
        {image && (
          <div className={`${flexCenterClasses} relative w-full h-full image-container`}>
            {image && (
              <img src='/images/close_button.png' className='close-button' onClick={clearImage1}/>
            )}
            <Tool handleMouseClick={handleMouseClick} handleMouseMove={handleMouseMove} />
          </div>
        )}
      </div>
    </div>
  );
};

export default Stage;
