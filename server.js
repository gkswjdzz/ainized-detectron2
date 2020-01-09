var http = require("http"),
  express = require("express"),
  Busboy = require("busboy"),
  path = require("path"),
  fs = require("fs"),
  inspect = require("util").inspect;
const { PythonShell } = require("python-shell");

var app = express();
var repo_dir = '/workspace/detectron2_repo';

var fullUrl = "",
  kind = "";

//return (호출한 알고리즘)
function busboyFunc(req, res, algorithm) {
  return new Promise((resolve, reject) => {
    let fileuploaded = true;
    var busboy = new Busboy({ headers: req.headers });
    
    busboy.on("file", function(fieldname, file, filename, encoding, mimetype) {
      if (filename === "") {
        fileuploaded = false;
      }
      file.pipe(fs.createWriteStream(__dirname + '/input_' + algorithm + '.jpg'));
    });

    busboy.on("field", function(fieldname, val, fieldnameTruncated, valTruncated, encoding, mimetype) {
      if(val === 'undefined')
        fileuploaded = false;
      kind = inspect(val).substring(1, inspect(val).length - 1);
    });

    busboy.on("finish", function() {
      if (!fileuploaded) {
        res.writeHead(400);
        res.end();
        return;
      }
      console.log("before resolve");
      resolve(kind);
    });
    req.pipe(busboy);
  }).then(function(kind){
    console.log("then");
    return [__dirname + '/input_' + kind + '.jpg', __dirname + '/output_' + kind + '.jpg'];
  })
}

app.get("/", function(req, res) {
  fullUrl = req.protocol + "://" + req.get("host") + req.originalUrl;

  res.writeHead(200, { "Content-Type": "text/html" });
  res.write('<form action="' + fullUrl + '" method="post" enctype="multipart/form-data">');
  res.write('<input type="file" accept="image/*" name="filetoupload"><br>');
  res.write('<input type="radio" name="kind" checked="checked" value="densepose" /> DensePose<br>');
  res.write('<input type="radio" name="kind" value="instancesegmentation" /> Instance Segementation<br>');
  res.write('<input type="radio" name="kind" value="panopticsegmentation" /> Panoptic Segmentation<br>');
  res.write('<input type="radio" name="kind" value="keypoint" /> Keypoint Detection<br>');
  res.write('<input type="submit">');
  res.write("</form>");
  
  return res.end();
});

app.post("/", async function(req, res){
  console.log("here")
  const ret = await busboyFunc(req, res);
  res.redirect(307, fullUrl + kind);
});

app.post("/densepose", async function(req, res) {
  const [newInput, newOutput] = await busboyFunc(req, res, 'densepose');
  const { i, o } = await runDensePosePython(newInput, newOutput);
  var s = fs.createReadStream(newOutput);
  s.on('open', function () {
    res.set('Content-Type', 'image/png');
    s.pipe(res);
  });
});

app.post("/panopticsegmentation", async function(req, res) {
  const [newInput, newOutput] = await busboyFunc(req, res, 'panopticsegmentation');
  config = 'panoptic_fpn_R_50_inference_acc_test.yaml';
  const { i, o } = await runPython(newInput, newOutput, config);
  var s = fs.createReadStream(newOutput);
  s.on('open', function () {
    res.set('Content-Type', 'image/png');
    s.pipe(res);
  });
});

app.post("/instancesegmentation", async function(req, res) {
  const [newInput, newOutput] = await busboyFunc(req, res, 'instancesegmentation');
  config = 'mask_rcnn_R_50_FPN_inference_acc_test.yaml';
  const { i, o } = await runPython(newInput, newOutput, config);
  var s = fs.createReadStream(newOutput);
  s.on('open', function () {
    res.set('Content-Type', 'image/png');
    s.pipe(res);
  });
});

app.post("/keypoint", async function(req, res) {
  const [newInput, newOutput] = await busboyFunc(req, res, 'keypoint');
  config = 'keypoint_rcnn_R_50_FPN_inference_acc_test.yaml';
  const { i, o } = await runPython(newInput, newOutput, config);
  var s = fs.createReadStream(newOutput);
  s.on('open', function () {
    res.set('Content-Type', 'image/png');
    s.pipe(res);
  });
});

app.listen(80, () => {
  console.log("server connect");
});

//run python except densepose
runPython = (input, output, config) => {
  return new Promise((resolve, reject) => {
    PythonShell.run(
      repo_dir + "/demo.py",
      { args: ["--input", input,
              "--output", output,
              "--config-file", repo_dir + "/configs/quick_schedules/" + config ] },
      async (err, result) => {
        if (err) {
          if (err.traceback === undefined) {
            console.log(err.message);
          } else {
            console.log(err.traceback);
          }
        }
        const inputdir = await result[result.length - 2];
        const outputdir = await result[result.length - 1];
        resolve({ inputdir, outputdir });
      }
    );
  });
};

// run densepose
runDensePosePython = (input, output) => {
  return new Promise((resolve, reject) => {
    PythonShell.run(
      repo_dir + "/apply_net.py",
      {
        args: [
          "show",
          repo_dir + "/configs/densepose_rcnn_R_50_FPN_s1x.yaml",
          repo_dir + "/densepose_rcnn_R_50_FPN_s1x.pkl",
          input,
          "dp_contour,bbox",
          "--output",
          output
        ]
      },
      async (err, result) => {
        if (err) {
          if (err.traceback === undefined) {
            console.log(err.message);
          } else {
            console.log(err.traceback);
          }
        }
        const inputdir = await result[result.length - 2];
        const outputdir = await result[result.length - 1];
        resolve({ inputdir, outputdir });
      }
    );
  });
};
