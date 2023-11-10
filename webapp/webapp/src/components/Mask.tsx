import React, { useContext } from 'react';
import { MaskProps } from "./helpers/Interfaces";
import LineChart from './LineChart';
import { Button, Checkbox } from "@material-tailwind/react";
import {
  TrashIcon
} from "@heroicons/react/24/outline";
import { ControlContext } from './hooks/createContext';

const Mask = ({ mask, index, handleDelete }: MaskProps) => {
  const {
    selectedIndices: [selectedIndices, setSelectedIndices]
  } = useContext(ControlContext)!;

  const handleChange = (event: any) => {
    console.log("vatran", event, index)
    if (event.target.checked) {
      addToSelectedIndices(index);
    } else {
      removeFromSelectedIndices(index);
    }
  }

  const addToSelectedIndices = (index: number) => {
    setSelectedIndices([...selectedIndices, index])
  }

  const removeFromSelectedIndices = (index: number) => {
    const indices = selectedIndices.filter((x) => x != index)
    setSelectedIndices(indices)
  }

  return (
    <div className="flex gap-3 items-center">
      <div>
      <Checkbox onChange={handleChange} checked={selectedIndices.indexOf(index) >= 0} onResize={undefined} onResizeCapture={undefined} crossOrigin={undefined}/>
      <Button className="flex items-center bg-red-500" onClick={() => { handleDelete(index); }} onResize={undefined} onResizeCapture={undefined}>
        <TrashIcon strokeWidth={2} className="h-5 w-5" />
      </Button>
      </div>
      <img src={mask.maskImg.src} alt="Photo" className="w-[30%] border-outline" />
      <div className="flex"><LineChart maskIndex={index}/></div>
    </div>
  );
}

export default Mask;