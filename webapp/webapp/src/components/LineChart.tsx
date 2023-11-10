
import React, { useState, useContext, useEffect, useRef } from "react";
import { ControlContext } from "./hooks/createContext";
import { Line } from 'react-chartjs-2';
import 'chartjs-plugin-dragdata';
import { Button } from "@material-tailwind/react";
import { LineChartProps } from "./helpers/Interfaces";
const _ = require('lodash');

const makeData = (size: number, data: Array<number> | null) => {
  let intArray;
  if (data == null) {
    intArray = [...Array(size).keys()];
    return {
      labels: intArray.map(x => `${x}`),
      datasets: [{
        data: intArray.map(x => x * 100.0 / (size - 1)),
        borderColor: '9B9B9B',
        borderWidth: 1,
        pointRadius: 10,
        pointHoverRadius: 10,
        pointBackgroundColor: '#609ACF',
        pointBorderWidth: 0,
        spanGaps: false,
      }],
    }
  } else {
    if (data.length < size) {
      intArray = [...data, 100];
    } else if (data.length > size) {
      data.pop()
      intArray = [...data];
    } else {
      intArray = [...data];
    }
    const labels = [...Array(size).keys()];
    return {
      labels: labels.map(x => `${x}`),
      datasets: [{
        data: intArray,
        borderColor: '9B9B9B',
        borderWidth: 1,
        pointRadius: 10,
        pointHoverRadius: 10,
        pointBackgroundColor: '#609ACF',
        pointBorderWidth: 0,
        spanGaps: false,
      }],
    }
  }
}

const LineChart = ({ maskIndex }: LineChartProps) => {
  const {
    controls: [controls, setControls]
  } = useContext(ControlContext)!;
  const [control, setControl] = useState(makeData(6, null));
  const [importing, setImporting] = useState(false);
  const [importValue, setImportValue] = useState('');

  // const importRef = useRef<HTMLInputElement>(null);
  
  useEffect(() => {
    console.log("vatran", control)
    const newControl = [...controls];
    newControl[maskIndex] = { points: control['datasets'][0].data };
    setControls(newControl);
  }, [control]);

  const options = {
    tooltips: { enabled: true },
    scales: {
      xAxes: [
        {
          gridLines: { display: false, color: "grey" },
          ticks: { fontColor: "#3C3C3C", fontSize: 14 }
        }
      ],
      yAxes: [
        {
          scaleLabel: {
            display: true,
            labelString: "Interpolation Value",
            fontSize: 14
          },
          ticks: {
            display: true,
            min: 0,
            max: 100,
            scaleSteps: 50,
            scaleStartValue: -50,
            maxTicksLimit: 4,
            fontColor: "#9B9B9B",
            padding: 30,
            // callback: point => (point < 0 ? "" : point)
          },
          gridLines: {
            display: false,
            offsetGridLines: true,
            color: "3C3C3C",
            tickMarkLength: 4
          }
        }
      ]
    },
    legend: {
      display: false
    },
    dragData: true,
    onDragStart: function(e: any) {
      console.log(e);
    },
    onDrag: function(e: any, datasetIndex: any, index: any, value: any) {
      // console.log(datasetIndex, index, value);
    },
    onDragEnd: function(e: any, datasetIndex: any, index: any, value: any) {
      console.log(datasetIndex, index, value);
      const points = [...control['datasets'][0].data];
      points[index] = value;
      console.log("vatran debug", "points", points);
      const tmpControl = makeData(points.length, null);
      tmpControl['datasets'][0].data = points;
      setControl(tmpControl);
    }
  };

  const handleClick = () => {
    const current = control['datasets'][0].data.length
    setControl(makeData(current, null));
  }

  const handleImport = () => {
    console.log("vatran import");
    // todo show input dialog
    setImporting(true);
  }

  const handleImportOk = () => {
    setImporting(false);
    const valuesString = importValue
    const controls = valuesString.split(",");
    if (controls.length == 0) return;
    
    setControl(makeData(controls.length, controls.map((x) => parseFloat(x))));
  }

  const decreaseValue = () => {
    const current = control['datasets'][0].data.length
    if (current <= 2) return;
    setControl(makeData(current - 1, control['datasets'][0].data));
  }

  const increaseValue = () => {
    const current = control['datasets'][0].data.length
    if (current >= 10) return;
    setControl(makeData(current + 1, control['datasets'][0].data));
  }

  const handleImportValueChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setImportValue(event.target.value);
  };

  function showNotImporting() {
    return  (
      <div className='flex gap-2 margin-vertical-20'>
        <form className="flex">
          <div className="value-button" id="decrease" onClick={decreaseValue}>-</div>
          <input type="number" id="number" value={control['datasets'][0].data.length} disabled={true} />
          <div className="value-button" id="increase" onClick={increaseValue}>+</div>
        </form>
        <Button size="sm" onClick={() => { handleClick(); } } onResize={undefined} onResizeCapture={undefined}>Default</Button>
        <Button size="sm" onClick={() => { handleImport(); }} onResize={undefined} onResizeCapture={undefined}>Import</Button>
      </div>
    )
  }

  function showImporting() {
    return  (
      <div className='flex gap-2 margin-vertical-20'>
        <input id="import" className="grow" placeholder="value1, v2, v3, v4, ..." value={importValue} onChange={handleImportValueChange}/>
        <Button size="sm" onClick={() => { handleImportOk(); }} disabled={!importValue} onResize={undefined} onResizeCapture={undefined}>Ok</Button>
      </div>
    )
  }

  return (
    <div>
      {
        importing ? 
        showImporting()
        : showNotImporting()
      }
      <Line data={control} options={options} redraw={false} />
    </div>
  );
};

export default LineChart;
