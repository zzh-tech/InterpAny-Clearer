// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.

// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import { Tensor } from "onnxruntime-web";

export interface modelRawMaskProps {
  height: number;
  width: number;
  data: Uint8ClampedArray;
}

export interface modelMaskProps {
  id: string;
  name: string;
  maskImg: HTMLImageElement;
  rawMask: modelRawMaskProps;
}

export interface modelControlProps {
  points: Array<number>
}

export interface modelScaleProps {
  samScale: number;
  height: number;
  width: number;
}

export interface modelInputProps {
  x: number;
  y: number;
  clickType: number;
}

export interface modeDataProps {
  clicks?: Array<modelInputProps>;
  tensor: Tensor;
  modelScale: modelScaleProps;
}

export interface StageAppProps {
  setDefaultRawMask: (rawMask: modelRawMaskProps | null) => void;
  addMask: (mask: modelMaskProps) => void;
  updateLoader: (loading: boolean) => void;
  onUploadImage1: (path: string) => void;
  onUploadImage2: (path: string) => void;
}

export interface StageProps {
  setDefaultRawMask: (rawMask: modelRawMaskProps | null) => void;
  addMask: (mask: modelMaskProps) => void;
}

export interface ControlProps {
  defaultRawMask: modelRawMaskProps | null;
  masks: Array<modelMaskProps>;
  image1Path: string;
  image2Path: string;
  handleDelete: (index: number) => void;
  updateMasks: (masks: Array<modelMaskProps>) => void;
  updateLoader: (loading: boolean) => void;
}

export interface ToolProps {
  handleMouseClick: (e: any) => void;
  handleMouseMove: (e: any) => void;
}

export interface MaskListProps {
  masks: Array<{id: string, name: string, maskImg: HTMLImageElement}>
  handleDelete: (index: number) => void;
}

export interface MaskProps {
  mask: {id: string, name: string, maskImg: HTMLImageElement};
  index: number;
  handleDelete: (index: number) => void;
}

export interface LineChartProps {
  maskIndex: number
}