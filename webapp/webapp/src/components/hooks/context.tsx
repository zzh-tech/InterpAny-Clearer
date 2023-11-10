// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.

// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import React, { useState } from "react";
import { modelInputProps, modelRawMaskProps, modelMaskProps, modelControlProps } from "../helpers/Interfaces";
import { AppContext, ControlContext } from "./createContext";

export const AppContextProvider = (props: {
  children: React.ReactElement<any, string | React.JSXElementConstructor<any>>;
}) => {
  const [clicks, setClicks] = useState<Array<modelInputProps> | null>(null);
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [rawMask, setRawMask] = useState<modelRawMaskProps | null>(null);
  const [maskImg, setMaskImg] = useState<HTMLImageElement | null>(null);

  return (
    <AppContext.Provider
      value={{
        clicks: [clicks, setClicks],
        image: [image, setImage],
        rawMask: [rawMask, setRawMask],
        maskImg: [maskImg, setMaskImg]
      }}
    >
      {props.children}
    </AppContext.Provider>
  );
};

export const ControlContextProvider = (props: {
  children: React.ReactElement<any, string | React.JSXElementConstructor<any>>;
}) => {
  
  const [controls, setControls] = useState<Array<modelControlProps>>([]);
  const [selectedIndices, setSelectedIndices] = useState<Array<number>>([]);

  return (
    <ControlContext.Provider
      value={{
        controls: [controls, setControls],
        selectedIndices: [selectedIndices, setSelectedIndices],
      }}
    >
      {props.children}
    </ControlContext.Provider>
  );
};
