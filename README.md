# colony-counter

A little helper I made while taking AP Biology.
It uses OpenCV to detect and count E. coli bacteria colonies.
The default settings are optimized for counting pGLO colonies under UV light.

**Disclaimer:** The supplied image, `test.jpg`, has been modified using image editing software to remove names written on the agar plate.

## Try it here

I'm currently self hosting Colony Counter for anyone to use here: [colony-counter.server.sigmondkukla.dev](https://colony-counter.server.sigmondkukla.dev)

## Usage

1. Ensure that a relatively recent version of Python 3 is installed
2. Run `pip install -r requirements.txt`
3. Run `streamlit run app.py` and your browser should open to the application
