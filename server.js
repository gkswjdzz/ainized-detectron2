var http = require("http"),
  express = require("express"),
  Busboy = require("busboy"),
  path = require("path"),
  fs = require("fs");

const { PythonShell } = require("python-shell");
var app = express();
var input = path.join(__dirname, "uploads/input.jpg");
var output = path.join(__dirname, "output.jpg");

app.get("/", function(req, res) {
  res.writeHead(200, { "Content-Type": "text/html" });
  res.write(
    '<form action="fileupload" method="post" enctype="multipart/form-data">'
  );
  res.write('<input type="file" name="filetoupload"><br>');
  res.write('<input type="submit">');
  res.write("</form>");
  return res.end();
});

app.post("/readfile", async (req, res) => {
  await runPython(input, output);
  console.log("start readfile");
  res.writeHead(200, { "Content-Type": "text/html" });
  res.write("<html><body>");

  console.log(output);

  //const file = () => {
    fs.readFile(output, (err, data) => {
      if (err) throw err;
      res.write('<img src="data:image/jpeg;base64,');
      res.write(Buffer.from(data).toString('base64'));
      res.write('" width="500" height="500" />');
      //res.write(filename);
      res.end("</body></html>");
    });
  //};

  //Promise.resolve(file, async () => {
  //});
  console.log("end readfile");
});

app.post("/fileupload", function(req, res) {
  var busboy = new Busboy({ headers: req.headers });
  busboy.on("file", function(fieldname, file, filename, encoding, mimetype) {
    file.pipe(fs.createWriteStream(input));
  });

  busboy.on('finish', function() {
    console.log('Upload complete');
    //res.writeHead(303, { 'Connection': 'close', Location : '/readfile' });
    //res.end();
    //res.end("That's all folks!");

    res.redirect(307, '/readfile');
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
      "./detectron2/demo.py", { args: [input, output] },
      async (err, result) => {
        if (err) {
          if (err.traceback === undefined) {
            console.log(err.message);
          } else {
            console.log(err.traceback);
          }
        }
        // const basePath = await result[result.length - 3];
        // const stylePath = await result[result.length - 2];
        // const outputPath = await result[result.length - 1];
        // console.log(basePath, stylePath, outputPath);
        // resolve({basePath, stylePath, outputPath});
        resolve();
      }
    );
  });
};