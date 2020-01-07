var http = require("http"),
  express = require("express"),
  Busboy = require("busboy"),
  path = require("path"),
  fs = require("fs"),
  inspect = require("util").inspect;
const { PythonShell } = require("python-shell");

var app = express();
var input = path.join(__dirname, "uploads/input.jpg");
var output = path.join(__dirname, "uploads/output.jpg");
var fullUrl = "",
  kind = "";

function busboyFunc(req, res) {
  return new Promise((resolve, reject) => {
    let fileuploaded = true;
    var busboy = new Busboy({ headers: req.headers });
    busboy.on("file", function(fieldname, file, filename, encoding, mimetype) {
      console.log(fieldname, filename, file);
      if (filename === "") {
        fileuploaded = false;
        console.log("here");
      }
      console.log(fieldname, filename);
      file.pipe(fs.createWriteStream(input));
    });

    busboy.on("field", function(
      fieldname,
      val,
      fieldnameTruncated,
      valTruncated,
      encoding,
      mimetype
    ) {
      console.log("Field [" + fieldname + "]: value: " + val);
      kind = inspect(val).substring(1, inspect(val).length - 1);
    });

    busboy.on("finish", function() {
      console.log("Upload complete");
      if (!fileuploaded) {
        res.writeHead(400);
        res.end();
        return;
      }

      resolve(kind);
    });

    req.pipe(busboy);
    console.log("end fileupload post");
  });
}

app.get("/", function(req, res) {
  fullUrl = req.protocol + "://" + req.get("host") + req.originalUrl;
  //console.log(req);
  console.log(req.headers.host);
  console.log(req.originalUrl);
  console.log(fullUrl + "densepose");

  res.writeHead(200, { "Content-Type": "text/html" });
  res.write(
    '<form action="' +
      fullUrl +
      '" method="post" enctype="multipart/form-data">'
  );
  res.write('<input type="file" accept="image/*" name="filetoupload"><br>');
  res.write(
    '<input type="radio" name="kind" checked="checked" value="densepose" /> DensePose'
  );
  res.write(
    '<input type="radio" name="kind" value="instancesegmentation" /> Instance Segementation'
  );
  res.write('<input type="submit">');
  res.write("</form>");
  return res.end();
});

app.post("/", async function(req, res){
  const ret = await busboyFunc(req, res);
  res.redirect(307, fullUrl + kind + 'Web');
});

app.post("/readfile", function(req, res) {
  res.writeHead(200, { "Content-Type": "text/html" });
  res.write("<html><body>");

  console.log(output);

  fs.readFile(output, (err, data) => {
    if (err) throw err;
    res.write('<img src="data:image/jpeg;base64,');
    res.write(Buffer.from(data).toString("base64"));
    res.write('"/>');
    res.end("</body></html>");
  });
  console.log("end readfile");
});

app.post("/densepose", async function(req, res) {
  const ret = await busboyFunc(req, res);
  console.log(input, output);
  const { i, o } = await runDensePosePython(input, output);

  console.log(i, o);
  console.log("start readfile");
  var s = fs.createReadStream(output);
  s.on('open', function () {
    res.set('Content-Type', 'image/png');
    s.pipe(res);
  });
  console.log(output);
  console.log("end readfile");
});

app.post("/instancesegmentation", async function(req, res) {
  console.log(input, output);
  const { i, o } = await runPython(input, output);

  console.log(i, o);
  console.log("start readfile");
  var s = fs.createReadStream(output);
  s.on('open', function () {
    res.set('Content-Type', 'image/png');
    s.pipe(res);
  });
  console.log(output);
  console.log("end readfile");
});

app.post("/denseposeWeb", async function(req, res) {
  console.log(input, output);
  const { i, o } = await runDensePosePython(input, output);

  console.log(i, o);
  console.log("start readfile");
  res.redirect(307, fullUrl + "readfile");
});

app.post("/instancesegmentationWeb", async function(req, res) {
  console.log(input, output);
  const { i, o } = await runPython(input, output);

  console.log(i, o);
  console.log("start readfile");
  res.redirect(307, fullUrl + "readfile");
});

app.listen(80, () => {
  console.log("server connect");
});

//run python except densepose
runPython = (input, output) => {
  return new Promise((resolve, reject) => {
    PythonShell.run(
      __dirname + "/detectron2_repo/demo.py",
      { args: ["--input", input] },
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
        console.log(inputdir, outputdir);
        resolve({ inputdir, outputdir });
      }
    );
  });
};

// run densepose
runDensePosePython = (input, output) => {
  return new Promise((resolve, reject) => {
    PythonShell.run(
      __dirname + "/detectron2_repo/apply_net.py",
      {
        args: [
          "show",
          "/workspace/detectron2_repo/configs/densepose_rcnn_R_50_FPN_s1x.yaml",
          "/workspace/densepose_rcnn_R_50_FPN_s1x.pkl",
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
        console.log(inputdir, outputdir);
        resolve({ inputdir, outputdir });
      }
    );
  });
};
