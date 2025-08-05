from flask import Flask, render_template_string

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Dota2Coach Overlay</title>
    <style>
        html, body {
            margin: 0;
            padding: 0;
            background: transparent;
            overflow: hidden;
            width: 100vw;
            height: 100vh;
        }

        #content {
            font-family: sans-serif;
            font-size: 28px;
            color: white;
            background-color: rgba(0, 0, 0, 0.7);
            padding: 16px 24px;
            border-radius: 12px;
            position: absolute;
            top: 100px;
            left: 100px;
            cursor: move;
            user-select: none;
        }
    </style>
</head>
<body>
    <div id="content">👋 Hello, Dota2Coach!</div>

    <script>
        const el = document.getElementById("content");
        let offsetX, offsetY, isDragging = false;

        el.addEventListener("mousedown", function(e) {
            isDragging = true;
            offsetX = e.clientX - el.offsetLeft;
            offsetY = e.clientY - el.offsetTop;
        });

        window.addEventListener("mouseup", function() {
            isDragging = false;
        });

        window.addEventListener("mousemove", function(e) {
            if (!isDragging) return;
            el.style.left = (e.clientX - offsetX) + "px";
            el.style.top = (e.clientY - offsetY) + "px";
        });
    </script>
</body>
</html>
"""


@app.route("/overlay")
def overlay():
    return render_template_string(HTML_TEMPLATE)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
