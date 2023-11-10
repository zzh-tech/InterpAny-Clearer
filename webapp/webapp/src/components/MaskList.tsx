import React from "react";
import { MaskListProps } from "./helpers/Interfaces";
import Mask from './Mask';

const MaskList = ({ masks, handleDelete }: MaskListProps) => {

  return (
    <>
      {
        masks.map(mask => {
          const index = masks.indexOf(mask);
          return <Mask key={mask.id} mask={mask} index={index} handleDelete={handleDelete} />
        })
      }
    </>
  );
}

export default MaskList;