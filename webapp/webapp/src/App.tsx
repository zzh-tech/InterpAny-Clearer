import React, { useState } from 'react';
import { AppContextProvider, ControlContextProvider } from './components/hooks/context';
import StageApp from './StageApp';
import ControlApp from './ControlApp';
import { modelMaskProps, modelRawMaskProps } from './components/helpers/Interfaces';

// dog
// const IMAGE_PATH1 = "/assets/data/dog.jpg";
// const IMAGE_PATH2 = "/assets/data/dog.jpg";
// const IMAGE_EMBEDDING = "/assets/data/dog_embedding.npy";

// dogs
// const IMAGE_PATH1 = "/assets/data/dogs.jpg";
// const IMAGE_PATH2 = "/assets/data/dogs.jpg";
// const IMAGE_EMBEDDING = "/assets/data/dogs_embedding.npy";

// truck
// const IMAGE_PATH1 = "/assets/data/truck.jpg";
// const IMAGE_PATH2 = "/assets/data/truck.jpg";
// const IMAGE_EMBEDDING = "/assets/data/truck_embedding.npy";

// groceries
const IMAGE_PATH1 = "/assets/data/groceries.jpg";
const IMAGE_PATH2 = "/assets/data/truck.jpg";
const IMAGE_EMBEDDING = "/assets/data/groceries_embedding_fast.npy";

// 000
// const IMAGE_PATH1 = "/assets/data/000.png";
// const IMAGE_PATH2 = "/assets/data/001.png";
// const IMAGE_EMBEDDING = "/assets/data/000_embedding.npy";

const App = () => {
  const [masks, setMasks] = useState<Array<modelMaskProps>>([]);
  const [defaultRawMask, setDefaultRawMask] = useState<modelRawMaskProps | null>(null);
  const [blocking, setBlocking] = useState<boolean>(false);
  const addMask = (mask: modelMaskProps) => {
    setMasks([...masks, mask]);
  }

  const [imagePath1, setImagePath1] = useState<string>("");
  const [imagePath2, setImagePath2] = useState<string>("");

  const handleDeleteMask = (index: number) => {
    const newMasks = [...masks];
    newMasks.splice(index,1)
    setMasks(newMasks);
  };

  const updateLoader = (loading: boolean) => {
    setBlocking(loading)
  }

  const updateMasks = (masks: Array<modelMaskProps>) => {
    console.log("vatran updateMasks", "")
    setMasks(masks);
  }

  const flexCenterClasses = "flex items-center justify-center m-auto";
  return (
    <>
      <div className="w-full h-full">
        <h1 className="text-center text-2xl my-3">Manipulated Interpolation of Anything</h1>
        <div className={`${flexCenterClasses} w-[90%] h-[90%]`}>
          <AppContextProvider>
            <StageApp setDefaultRawMask={setDefaultRawMask} addMask={addMask} updateLoader={updateLoader} onUploadImage1={setImagePath1} onUploadImage2={setImagePath2} />
          </AppContextProvider>
          <ControlContextProvider>
            <ControlApp defaultRawMask={defaultRawMask} masks={masks} handleDelete={handleDeleteMask} updateMasks={updateMasks} updateLoader={updateLoader}
              image1Path={imagePath1} image2Path={imagePath2} />
          </ControlContextProvider>
        </div>
      </div>
      
      <div className={blocking? 'flex model' : 'hidden' }>
        <div className='modal-content'>
          <div className='loader'></div>
          <div className='modal-text'>Loading...</div>
        </div>
      </div>
    </>
  );
}

export default App;