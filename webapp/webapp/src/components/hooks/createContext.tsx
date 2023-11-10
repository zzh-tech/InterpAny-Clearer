// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.

// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import { createContext } from "react";
import { modelInputProps, modelRawMaskProps, modelControlProps } from "../helpers/Interfaces";

interface contextProps {
  clicks: [
    clicks: modelInputProps[] | null,
    setClicks: (e: modelInputProps[] | null) => void
  ];
  image: [
    image: HTMLImageElement | null,
    setImage: (e: HTMLImageElement | null) => void
  ];
  rawMask: [
    rawMask: modelRawMaskProps | null, 
    setRawMask: (e: modelRawMaskProps | null) => void
  ],
  maskImg: [
    maskImg: HTMLImageElement | null,
    setMaskImg: (e: HTMLImageElement | null) => void
  ];
}

interface contextControlsProps {
  controls: [
    controls: Array<modelControlProps>,
    setControls: (e: Array<modelControlProps>) => void
  ];
  selectedIndices: [
    selectedIndices: Array<number>,
    setSelectedIndices: (e: Array<number>) => void
  ];
}

export const AppContext = createContext<contextProps | null>(null);
export const ControlContext = createContext<contextControlsProps | null>(null);
