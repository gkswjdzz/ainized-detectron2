var http = require("http"),
  express = require("express"),
  cors = require("cors"),
  Busboy = require("busboy"),
  path = require("path"),
  fs = require("fs"),
  inspect = require("util").inspect;
  uuidv4 = require('uuid/v4');

const { spawn } = require('child_process');

var app = express();
app.use(cors({
  origin: 'https://ainize.ai',
}));
var repo_dir = '/workspace/detectron2_repo';

var fullUrl = "",
  kind = "";

//return (호출한 알고리즘)
function busboyFunc(req, res) {
  return new Promise((resolve, reject) => {
    let fileuploaded = true;
    var busboy = new Busboy({ headers: req.headers });
    uuid4 = uuidv4();
    busboy.on("file", function(fieldname, file, filename, encoding, mimetype) {
      if (filename === "") {
        fileuploaded = false;
      }
      file.pipe(fs.createWriteStream(__dirname + '/input_' + uuid4 + '.jpg'));
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
      resolve(uuid4);
    });
    
    req.pipe(busboy);
  }).then(function(uuid4){
    console.log("then " + uuid4);
    return [__dirname + '/input_' + uuid4 + '.jpg', __dirname + '/output_' + uuid4 + '.jpg'];
  })
}

app.get("/", function(req, res) {
  fullUrl = req.protocol + "://" + req.get("host") + req.originalUrl;

  res.writeHead(200, { "Content-Type": "text/html" });
  res.write('<form action="' + fullUrl + '" method="post" enctype="multipart/form-data">');
  res.write('<input type="file" accept="image/*" name="filetoupload"><br>');
  res.write('<input type="radio" name="kind" checked="checked" value="densepose" /> DensePose<br>');
  res.write('<input type="radio" name="kind" value="instancesegmentation" /> Instance Segmentation<br>');
  res.write('<input type="radio" name="kind" value="panopticsegmentation" /> Panoptic Segmentation<br>');
  res.write('<input type="radio" name="kind" value="keypoint" /> Keypoint Detection<br>');
  res.write('<input type="submit">');
  res.write("</form>");
  
  return res.end();
});

app.post("/", async function(req, res){
  const ret = await busboyFunc(req, res, uuidv4());
  res.redirect(307, fullUrl + kind);
});

app.post("/densepose", async function(req, res) {
  const [newInput, newOutput] = await busboyFunc(req, res);
  await runDensePosePython(newInput, newOutput, res);
});

app.post("/panopticsegmentation", async function(req, res) {
  const [newInput, newOutput] = await busboyFunc(req, res);
  config = 'panoptic_fpn_R_50_inference_acc_test.yaml';
  runPython(newInput, newOutput, config, res);
});

app.post("/instancesegmentation", async function(req, res) {
  const [newInput, newOutput] = await busboyFunc(req, res);
  config = 'mask_rcnn_R_50_FPN_inference_acc_test.yaml';
  runPython(newInput, newOutput, config, res);
});

app.post("/keypoint", async function(req, res) {
  const [newInput, newOutput] = await busboyFunc(req, res);
  config = 'keypoint_rcnn_R_50_FPN_inference_acc_test.yaml';
  runPython(newInput, newOutput, config, res);
});

app.listen(80, () => {
  console.log("server connect");
});

//run python except densepose
runPython = (input, output, config, res) => {
  const pyProg = spawn('python', 
    [repo_dir + "/demo.py", 
      "--input", input,
      "--output", output,
      "--config-file", repo_dir + "/configs/quick_schedules/" + config ]);
  pyProg.stdout.on('data', function(data) {
    console.log('runPython func stdout : ' + data.toString());
  });
  pyProg.stderr.on('data', function(data) {
    console.log('runPython func stderr : ' + data.toString()); 
  });
  pyProg.on('close', (code) => {
    console.log('runPython exit code : ' + code);
    var s = fs.createReadStream(output);
    s.on('open', function () {
      res.set('Content-Type', 'image/png');
      s.pipe(res);
    });
  })
};

// run densepose
runDensePosePython = (input, output, res) => {
  const pyProg = spawn('python', 
    [
      repo_dir + "/apply_net.py", 
      "show",
      repo_dir + "/configs/densepose_rcnn_R_50_FPN_s1x_legacy.yaml",
      repo_dir + "/densepose_rcnn_R_50_FPN_s1x.pkl",
      input,
      "dp_contour,bbox",
      "--output",
      output 
    ]);
  pyProg.stdout.on('data', function(data) {
    console.log('runPython func stdout : ' + data.toString());
  });
  pyProg.stderr.on('data', function(data) {
    console.log('runPython func stderr : ' + data.toString()); 
  });
  pyProg.on('close', (code) => {
    console.log('Densepose exit code : ' + code);
    var s = fs.createReadStream(output);
    s.on('open', function () {
      res.set('Content-Type', 'image/png');
      s.pipe(res);
    });
  })
};
