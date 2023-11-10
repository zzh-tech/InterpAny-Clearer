import React, { useState, useContext, useEffect, useRef } from "react";
const axios = require("axios");
const SERVER_URL = "http://127.0.0.1:5001";

// drag drop file component
function DragDropFile({ message, uploadedCallback }) {
    // drag state
    const [dragActive, setDragActive] = React.useState(false);
    // ref
    const inputRef = React.useRef(null);
    
    // handle drag events
    const handleDrag = function(e) {     
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    };

    // triggers when file is dropped
    const handleDrop = function(e) {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        // at least one file has been dropped so do something
        // handleFiles(e.dataTransfer.files);
            console.log(e.dataTransfer.files[0]);
            uploadFile(e.dataTransfer.files[0]);
        }
    };

    // triggers when file is selected with click
    const handleChange = function(e) {
        e.preventDefault();
        if (e.target.files && e.target.files[0]) {
        // handleFiles(e.target.files);
            console.log(e.target.files[0]);
            uploadFile(e.target.files[0]);
        }
    };

    // triggers the input when the button is clicked
    const onButtonClick = () => {
        inputRef.current.click();
    };

    const uploadFile = (file) => {
        let formData = new FormData();
        
        formData.append("file", file);
    
        axios.post(`${SERVER_URL}/upload`, formData, {
            headers: {
                "Content-Type": "multipart/form-data",
            },
        })
        .then((response) => {
            console.log(response);
            const url = new URL(response['data']['url'], SERVER_URL);
            const fileName = response['data']['filename'];
            uploadedCallback(url, fileName);
        }, (error) => {
            console.log(error);
        });
    };
    
    return (
        <form id="form-file-upload" onDragEnter={handleDrag} onSubmit={(e) => e.preventDefault()}>
            <input ref={inputRef} type="file" id="input-file-upload" multiple={true} onChange={handleChange} />
            <label id="label-file-upload" htmlFor="input-file-upload" className={dragActive ? "drag-active" : "" }>
                <div>
                    <p>{message}</p>
                    <button className="upload-button" onClick={onButtonClick}>Upload a file</button>
                </div> 
            </label>
            { dragActive && <div id="drag-file-element" onDragEnter={handleDrag} onDragLeave={handleDrag} onDragOver={handleDrag} onDrop={handleDrop}></div> }
        </form>
    );
  };

  export default DragDropFile;