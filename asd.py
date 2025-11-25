from http.server import BaseHTTPRequestHandler, HTTPServer

class MyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        message = "Sdeneme"
        self.send_response(200)
        self.send_header("Content-type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(message.encode("utf-8"))

if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 8080), MyHandler)
    print("Server 8080 portunda çalışıyor...")
    server.serve_forever()
