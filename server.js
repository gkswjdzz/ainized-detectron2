var http = require("http"),
  express = require("express"),
  Busboy = require("busboy"),
  path = require("path"),
  fs = require("fs");

var app = express();

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

app.post("/readfile", (req, res) => {

  res.writeHead(200, { "Content-Type": "text/html" });
  res.write("<html><body>");

  filename = "output.jpg";

  const files = () => {
    fs.readFile(filename, (err, data) => {
      if (err) throw err;
      res.write('<img src="data:image/jpeg;base64,');
      res.write(Buffer.from(data).toString("base64"));
      res.write(" />");
    });
  };

  Promise.all(files, async () => {
    res.end("</body></html>");
  });
});

app.post("/fileupload", function(req, res) {
  var busboy = new Busboy({ headers: req.headers });
  busboy.on("file", function(fieldname, file, filename, encoding, mimetype) {
    var saveTo = path.join(__dirname, "uploads/" + filename);
    file.pipe(fs.createWriteStream(saveTo));
  });

  busboy.on('finish', function() {
    console.log('Upload complete');
    res.writeHead(200, { 'Connection': 'close' });
    res.end("That's all folks!");
  });
  return req.pipe(busboy);

});

app.listen(80, () => {
  console.log("server connect");
});
