var http = require("http"),
  express = require("express"),
  Busboy = require("busboy"),
  path = require("path"),
  fs = require("fs");

const { PythonShell } = require("python-shell");
var app = express();
var input = path.join(__dirname, "uploads/input.jpg");
var output = path.join(__dirname, "output.jpg");
var fullUrl = "";

app.get("/", function(req, res) {
  fullUrl = req.protocol + "://" + req.get("host") + req.originalUrl;
  console.log(fullUrl);
  res.writeHead(200, { "Content-Type": "text/html" });
  res.write(
    '<form action="' +
      fullUrl +
      'fileupload" method="post" enctype="multipart/form-data">'
  );
  res.write('<input type="file" name="filetoupload"><br>');
  res.write('<input type="submit">');
  res.write("</form>");
  return res.end();
});

app.post("/readfile", async (req, res) => {
  console.log(input, output);
  const { i, o } = await runPython(input, output);
  console.log(i, o);
  console.log("start readfile");
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

app.post("/fileupload", function(req, res) {
  var fileuploaded = true;
  var busboy = new Busboy({ headers: req.headers });
  busboy.on("file", function(fieldname, file, filename, encoding, mimetype) {
    if(filename === ""){
      fileuploaded = false;
      console.log("here");
    }
    console.log(fieldname, filename);
    file.pipe(fs.createWriteStream(input));
  });

  busboy.on("finish", function() {
    console.log("Upload complete");
    if(!fileuploaded) {
      res.writeHead(400);
      res.end();
      return;
    }

    res.redirect(307, fullUrl + "readfile");
  });

  req.pipe(busboy);
  console.log("end fileupload post");
});

app.listen(80, () => {
  console.log("server connect");
});

runPython = (input, output) => {
  return new Promise((resolve, reject) => {
    PythonShell.run(
      __dirname + "/detectron2_repo/demo.py",
      { args: [input, output] },
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
