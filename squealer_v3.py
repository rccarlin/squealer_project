import reactpy
import plotly
import plotly.io as pio
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from reactpy import component, html, hooks
from reactpy.backend.fastapi import configure
import uvicorn
import plotly.graph_objs as go
import json
import math
from sklearn.linear_model import LogisticRegression
import random
import csv
import io

# global variables for storing the current dataset, model mode, and the original best-fit line
global data
global model
global og_fit_line
data = None
model = None
og_fit_line = None
app = FastAPI()

# takes in the data (including handle bars), finds the line/ coefficients defined by the handle bars
# also returns the "residuals" which are residuals for linear regression but are signed margins for logistic regression
def line_fit(points, is_linear):

    if is_linear:
        x_all = points[0]
        y_all = points[1]

        x_main = x_all[:-2]
        y_main = y_all[:-2]

        right = (x_all[-1], y_all[-1])
        left  = (x_all[-2], y_all[-2])

        slope = (right[1] - left[1]) / (right[0] - left[0])
        intercept = right[1] - slope * right[0]
        coef = [slope, intercept]

        # compute the fitted values and residuals on the main data.
        fit_line = coef[0] * x_all + coef[1]
        resids = y_all - fit_line
        return fit_line, resids, coef
    
    # logistic mode
    else:
        # points = [x_all, y_all] (1-d x, probability y)
        x_all = points[0]
        y_all = points[1]

        # remove handlebar points
        x_main = x_all[:-2]
        y_main = y_all[:-2]

        # define the logistic curve from handlebars
        x1, x2 = x_all[-2], x_all[-1]
        y1, y2 = y_all[-2], y_all[-1]

        # compute logistic parameters using the logit transform
        beta1 = (logit(y2) - logit(y1)) / (x2 - x1)
        beta0 = logit(y1) - beta1 * x1

        # generate a set of x-values over the main data range
        x_line = np.linspace(np.min(x_main), np.max(x_main), 100)
        y_line = 1/(1+np.exp(-(beta0 + beta1*x_line)))
        fit_line = [x_line, y_line]

        # compute residuals as the difference between observed probabilities and predicted probabilities
        pred_main = 1/(1+np.exp(-(beta0 + beta1*x_main)))
        resids = y_main - pred_main
        coef = [beta0, beta1]
        return fit_line, resids, coef



# takes in all data (with handle bars at the end) to plot the data points, handle bars, original best fit line, and the
# current line defined by the handlebars
def make_plot(data, is_linear):

    # linear mode:
    if is_linear:
        x_all = data[0]
        y_all = data[1]
        x_main = x_all[:-2]
        y_main = y_all[:-2]
        fit_line, resids, coef = line_fit(data, is_linear)

        # convert to lists for plotly compatibility
        x_main_list = x_main.tolist()
        y_main_list = y_main.tolist()
        fit_line_list = fit_line.tolist()
        color_list = (resids**2).tolist()
        scatter_trace = go.Scatter(
            x=x_main_list,
            y=y_main_list,
            mode='markers',
            marker=dict(
                color=color_list,
                colorscale="portland",
                colorbar=dict(title="Residual Squared", x=1.1, y=0.5, len=0.5)
            ),
            name="Data Points"
        )
        curr_fit_trace = go.Scatter(
            x=x_main_list,
            y=fit_line_list,
            mode='lines',
            name='Current Fit Line'
        )
        # take out the two handlebar points
        handlebars_trace = go.Scatter(
            x=[x_all[-1], x_all[-2]],
            y=[y_all[-1], y_all[-2]],
            mode='markers',
            name='Handlebars',
            marker=dict(color='red', size=10)
        )
        # plot the original best-fit line based on handlebars (this line is fixed)
        og_fit_trace = go.Scatter(
            x=x_main_list,
            y=og_fit_line.tolist() if isinstance(og_fit_line, np.ndarray) else og_fit_line,
            mode='lines',
            name='Original Fit Line'
        )

        fig = go.Figure(data=[scatter_trace, curr_fit_trace, og_fit_trace, handlebars_trace])
        fig.update_layout(title="Data")
        return json.loads(fig.to_json())
    
    # logistic 
    else:
        # data = [x_all, y_all], y are probs
        x_all = data[0]
        y_all = data[1]

        # data (without handlebars)
        x_main = x_all[:-2]
        y_main = y_all[:-2]
        fit_line, margins, coef = line_fit(data, is_linear)
        scatter_trace = go.Scatter(
            x=x_main.tolist(),
            y=y_main.tolist(),
            mode='markers',
            marker=dict(
                color=margins.tolist(),
                colorscale="portland",
                colorbar=dict(title="Residual", x=1.1, y=0.5, len=0.5)
            ),
            name="Data Points"
        )
        curr_fit_trace = go.Scatter(
            x=fit_line[0].tolist(),
            y=fit_line[1].tolist(),
            mode='lines',
            name='Current Logistic Curve'
        )
        # plot the og logistic curve  (og_fit_line)
        og_fit_trace = go.Scatter(
            x=og_fit_line[0].tolist(),
            y=og_fit_line[1].tolist(),
            mode='lines',
            name='Original Logistic Curve'
        )

        # takes out the handlebars
        handlebars_trace = go.Scatter(
            x=[x_all[-1], x_all[-2]],
            y=[y_all[-1], y_all[-2]],
            mode='markers',
            name='Handlebars',
            marker=dict(color='red', size=10)
        )
        fig = go.Figure(data=[scatter_trace, curr_fit_trace, og_fit_trace, handlebars_trace])
        fig.update_layout(title="Data")
        return json.loads(fig.to_json())

# plots the intercepts and slopes of the model/ lines attempted so far, color is determined by the model's log likelihood
def make_prog_chart(coef_history, likelihood_history):
    size_list = [10] * len(likelihood_history)
    size_list[-1] = 25
    trace = go.Scatter(
        x=coef_history[1],
        y=coef_history[0],
        mode="lines+markers",
        marker=dict(
            size=size_list,
            color=likelihood_history,
            colorscale="Viridis",
            colorbar=dict(title="Approx Log Likelihood", x=1.1, y=0.5, len=0.5)
        ),
        name="Tries"
    )
    layout = go.Layout(
        title="Coefficients Tried (Current Try Larger)",
        xaxis_title="Intercept",
        yaxis_title="Slope"
    )
    fig = go.Figure(data=[trace], layout=layout)
    return json.loads(fig.to_json())

# calculates the log likelihood of the current model using the residuals (linear) or margins (logistic)
def log_likelihood(resids, is_linear):
    if is_linear:
        resid_squared = resids**2
        rss = resid_squared.sum()
        resid_var = np.var(resids, ddof=2)
        n = len(resids)
        return -n/2 * np.log(2*math.pi*resid_var) - rss/(2*resid_var)
    else:
        return np.sum(-np.log(1+np.exp(-resids)))


# plays a short tone at the given frequency (which will be some function of residuals/ margins)
def play_tone(err_val):
    maxErr = 100
    quality = math.exp(-err_val / maxErr)
    quality = max(0, min(quality, 1))
    base_freq = 880

    # "harmonic" chord for good fits (perfect fifth and octave)
    harmonic_intervals = [1.0, 1.5, 2.0]  

    # "dissonant" chord for poor fits (less consonant intervals)
    dissonant_intervals = [1.0, 1.1, 1.2] 

    # something in between the two sets based on quality
    intervals = []
    for h, d in zip(harmonic_intervals, dissonant_intervals):

        # quality=1, choose h; quality=0, choose d.
        intervals.append(d + (h - d) * quality)
    
    # calculate the frequencies for each oscillator
    freqs = [base_freq * mult for mult in intervals]

    # duration of the tone in milliseconds
    duration_ms = 200

    # build js code that creates multiple oscillators to play the chord simultaneously
    js = f"""
    (function() {{
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const now = audioCtx.currentTime;
        const duration = {duration_ms} / 1000;
        
        // Create a gain node to control overall volume
        const gainNode = audioCtx.createGain();
        gainNode.gain.value = 0.2;
        gainNode.connect(audioCtx.destination);
        
        // Frequencies for our chord:
        const freqs = {json.dumps(freqs)};
        
        freqs.forEach(freq => {{
            const osc = audioCtx.createOscillator();
            osc.type = 'sine';
            osc.frequency.value = freq;
            osc.connect(gainNode);
            osc.start(now);
            osc.stop(now + duration);
        }});
    }})();
    """
    return html.script(js)

# this component facilitates the plotting, updating, and replotting of data. Events such as key presses, sliders,
# and tones played are also handled here
@component
def InteractiveGraph():
    global data, model, og_fit_line

    # declare states (akin to global variables but for the UI specifically)
    points, set_points = hooks.use_state(data if data is not None else [np.array([]), np.array([])])
    pitch, set_pitch = hooks.use_state(0)
    is_linear, set_is_linear = hooks.use_state(model if model is not None else True)
    tries, set_tries = hooks.use_state([[0], [0]])
    likelihood_history, set_likelihood = hooks.use_state([0])
    step, set_step = hooks.use_state(0.10)

    # compute current best fit line from handlebars
    fit_line, resids, coef = line_fit(points, is_linear)
    # graphs
    graph_json = make_plot(points, is_linear)
    prog_chart_json = make_prog_chart(tries, likelihood_history)

    # this script is the html to actually display the graphs and prepare for key presses
    script = html.script(f"""
        function renderPlot() {{
            if (typeof Plotly !== "undefined") {{
                Plotly.newPlot('plot1', {json.dumps(graph_json['data'])}, {json.dumps(graph_json['layout'])});
                Plotly.newPlot('plot2', {json.dumps(prog_chart_json['data'])}, {json.dumps(prog_chart_json['layout'])});
            }} else {{
                console.error("Plotly not loaded");
            }}
        }}
        var plotlyScript = document.createElement('script');
        plotlyScript.src = 'https://cdn.plot.ly/plotly-latest.min.js';
        plotlyScript.onload = renderPlot;
        document.head.appendChild(plotlyScript);
    """)


    # add file input elements and upload button 
    file_upload_ui = html.div([
        html.div("Upload your CSV file:"),
        html.input({"type": "file", "id": "csvFileInput"}),
        html.button({"id": "uploadCsvButton"}, "Upload CSV"),
        html.div({"id": "uploadMsg"})
    ])
    
    # js to handle the file upload
    upload_script = html.script(f"""
    (function(){{
    const fileInput = document.getElementById('csvFileInput');
    const uploadButton = document.getElementById('uploadCsvButton');
    const uploadMsg = document.getElementById('uploadMsg');
    uploadButton.addEventListener('click', function(){{
        if (fileInput.files.length === 0){{
            uploadMsg.innerText = 'Please select a file first.';
            return;
        }}
        const file = fileInput.files[0];
        const formData = new FormData();
        formData.append('file', file);
        // Use the current mode from the ReactPy state.
        formData.append('mode', JSON.stringify({str(is_linear).lower()}));
        fetch('/upload_csv', {{
            method: 'POST',
            body: formData
        }})
        .then(response => {{
            if (!response.ok) throw new Error("Upload failed");
            return response.json();
        }})
        .then(data => {{
            uploadMsg.innerText = 'Upload successful! Reloading data...';
            setTimeout(() => window.location.reload(), 1000);
        }})
        .catch(error => {{
            console.error(error);
            uploadMsg.innerText = 'Upload error: ' + error;
        }});
    }});
}})();
""")

    # takes in the points, the index  (from the end) of the handlebar being updated, and the target handlebar's change
    # in x and y (or x1 and x2, in the case of logistic regression)
    # this could probably be rewritten to not need all points passed in, but this way has worked so far
    def update_point(all_points, index, dx, dy):

        new_points = [np.copy(arr) for arr in all_points]
        target_idx = -index

        # we do not want handlebars to detach from line/curve so we will
        # restrict them to the same boundaries as data (or as line/curve)
        
        # find the valid x range from main data 
        main_x = new_points[0][:-2]
        x_min, x_max = np.min(main_x), np.max(main_x)

        # update the chosen handlebar’s x and y by dx, dy and restrict them to domain
        new_points[0][target_idx] += dx

        # clamp x to the domain of the main data
        new_points[0][target_idx] = np.clip(new_points[0][target_idx], x_min, x_max)
        new_points[1][target_idx] += dy

        if is_linear:

            # restrict y to range of y values in data
            main_y = new_points[1][:-2]
            y_min, y_max = np.min(main_y), np.max(main_y)
            new_points[1][target_idx] = np.clip(new_points[1][target_idx], y_min, y_max)

        else:
            # restrict curve from 0 to 1 (since logistic)
            new_points[1][target_idx] = np.clip(new_points[1][target_idx], 0, 1)

        # recompute the best-fit line or curve with the updated data
        fit_line_new, resids_new, coef_new = line_fit(new_points, is_linear)

        # compute error for audio feedback
        if is_linear:
            temp = resids_new
            maxErr = 3500 
            err_val = np.sum(temp**2)
        else:
            temp = np.abs(resids_new)
            scale_factor = 100
            maxErr = 100   # what is a reasonable error?
            err_val = np.sum(temp**2) * scale_factor
        
        if err_val > maxErr:
            err_val = maxErr

        return new_points, err_val, coef_new, resids_new

    # Looks for arrow key or wasd presses and calls update_point() accordingly
    def handle_key_down(event):
        nonlocal pitch

        pressed = False
        index = 0 # will either have a value of 1 (left handlebar) or 2 (right handlebar)
        dx = 0
        dy = 0
        delta = float(step) # makes it so the amount changed (dx or dy) is determined by the step size, which is
        key = event["key"]
        
        # for each of the valid key presses, sets index and change in x or y (never both) to intended value
        # also sets pressed to True so that and update and redraw only happens when a handlebar is moved
        if key == "ArrowUp":
            index = 2
            dy = delta
            pressed = True
        elif key == "ArrowDown":
            index = 2
            dy = -delta
            pressed = True
        elif key == "ArrowLeft":
            index = 2
            dx = -delta
            pressed = True
        elif key == "ArrowRight":
            index = 2
            dx = delta
            pressed = True
        elif key == "w":
            index = 1
            dy = delta
            pressed = True
        elif key == "s":
            index = 1
            dy = -delta
            pressed = True
        elif key == "a":
            index = 1
            dx = -delta
            pressed = True
        elif key == "d":
            index = 1
            dx = delta
            pressed = True

        if pressed:
            # updates points, the tries, and the pitch
            new_points, new_err, new_coef, new_resids = update_point(points, index, dx, dy)
            set_points(new_points)
            new_tries = [tries[0] + [new_coef[0]], tries[1] + [new_coef[1]]]
            set_tries(new_tries)
            new_likelihood = likelihood_history + [log_likelihood(new_resids, is_linear)] # again, only need the labels for logistic, unused for linear
            set_likelihood(new_likelihood)
            set_pitch(new_err)


    # updates the step size according to changes with the slider
    def handle_slider_change(event):
        set_step(event["target"]["value"])

    # this function resets the data, tone, model attempts, etc. as it switches to/ from a linear/ logistic model    
    def switch_modes(event=None):
        set_is_linear(lambda prev: not prev)
        temp_mode = not is_linear  # new mode
        new_data = generate_data(temp_mode)
        global data
        data = new_data
        set_points(new_data)
        fit_line_new, resids_new, coef_new = line_fit(new_data, temp_mode)
        global og_fit_line

        # linear
        if temp_mode:  
            x_arr = new_data[0][:-2]
            y_arr = new_data[1][:-2]
            og_fit_line = np.polyval(np.polyfit(x_arr, y_arr, 1), x_arr)
        
        # logistic
        else:
            og_fit_line = fit_line_new

        set_pitch(0)
        set_likelihood([log_likelihood(resids_new, temp_mode)])

        # update coefficient tracking
        set_tries([[coef_new[0]], [coef_new[1]]])

    # the component returns the html necessary to display the graphs, handle key presses and sliders, and play tone
    return html.div([
        file_upload_ui,
        upload_script,
        html.div(
            {"style": {"display": "flex", "gap": "20px", "alignItems": "flex-start"}},
            [
                html.div({"id": "plot1", "style": {"width": "800px", "height": "600px"}}),
                html.div({"id": "plot2", "style": {"width": "600px", "height": "400px"}}),
            ],
        ),
        script,
        html.div({
            "tabIndex": 0,
            "autofocus": True,
            "onKeyDown": handle_key_down,
            "style": {"border": "1px solid black", "padding": "10px", "width": "300px"}
        }, "Use ARROW KEYS (right handlebar) and WASD (left handlebar) to move the handlebars independently."),
        html.div({"style": {"display": "flex", "alignItems": "center", "gap": "10px"}},
            [
                "Slider: ",
                html.input({
                    "type": "range",
                    "min": 0,
                    "max": 10,
                    "value": step,
                    "step": 0.10,
                    "onInput": handle_slider_change,
                }),
                html.span(f"Value: {step}")
            ]
        ),
        html.button({
            "onClick": switch_modes,
            "style": {"fontSize": "20px"}
        }, "Linear" if not is_linear else "Logistic"),
        play_tone(pitch)
    ])

# returns random (x, y) data for linear regression and random (x1, x2, y) data for logistic regression
def generate_data(is_linear):
    if is_linear:
        x = np.random.uniform(0, 10, 30)
        m = np.random.uniform(-3, 3)
        y = m * x + np.random.uniform(-3, 3, 30)

        # compute the original best-fit line from the main data.
        coef = np.polyfit(x, y, 1)
        slope = coef[0]
        intercept = coef[1]
        top = np.percentile(x, 80)
        bottom = np.percentile(x, 20)

        # append the two handlebar points computed from the original line
        x_full = np.append(x, [top, bottom])
        y_full = np.append(y, [slope * top + intercept, slope * bottom + intercept])
        return [x_full, y_full]
   
   # logistic mode data
    else:
        x = np.random.uniform(0, 10, 30)
        true_beta0 = -5
        true_beta1 = 1

        # compute probs using logistic model: these are ideal probabilities
        # from the true logistic function
        probs = 1 / (1 + np.exp(-(true_beta0 + true_beta1 * x)))

        # add some noise so the data does not lie exactly on the logistic curve
        noise_std  = 0.1 
        noisy_probs = probs + np.random.normal(0, noise_std, len(x))

        # we still want to make sure the data is between 0 and 1 though
        noisy_probs = np.clip(noisy_probs, 0, 1) 

        # get the baseline logistic regression by converting noisy probs to binary
        X = x.reshape(-1,1)
        y_bin = (noisy_probs >= 0.5).astype(int)
        mod = LogisticRegression()
        mod.fit(X, y_bin)

        # compute fitted probabilities from logistic regression model
        y_fit = 1 / (1 + np.exp(-(mod.intercept_[0] + mod.coef_[0][0] * x)))

        # choose handlebars at the 20th and 80th percentiles
        bottom = np.percentile(x, 20)
        top = np.percentile(x, 80)

        # compute the handlebar y-values on the true logistic curve
        y_bottom = 1 / (1 + np.exp(-(mod.intercept_[0] + mod.coef_[0][0] * bottom)))
        y_top = 1 / (1 + np.exp(-(mod.intercept_[0] + mod.coef_[0][0] * top)))

        # append the handlebar points in increasing x order
        x_full = np.append(x, [top, bottom])
        y_full = np.append(noisy_probs, [y_top, y_bottom])
        return [x_full, y_full]

def logit(p):
    # no division by 0! so we clip to fit within range for logistic regression
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.log(p/(1-p))

# for user inputted data via csv file
@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...), mode: bool = Form(...)):
    global data, og_fit_line, model
    model = mode

    """
    Expects a CSV file with two columns: x and y.
    Returns a JSON object with keys 'x' and 'y', where the arrays include
    the main data plus two added handlebar points.
    The 'mode' parameter indicates whether to prepare data for linear (True)
    or logistic (False) mode.
    """
    contents = await file.read()
    try:
        text = contents.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File encoding error")
    
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    
    # skip a header row if present
    if len(rows) > 0 and rows[0][0].lower() in ['x', 'x_value']:
        rows = rows[1:]

    x_vals = []
    y_vals = []
    for row in rows:
        if len(row) >= 2:
            try:
                x_vals.append(float(row[0]))
                y_vals.append(float(row[1]))
            except ValueError:
                continue

    if len(x_vals) == 0:
        raise HTTPException(status_code=400, detail="No valid data found")
    
    # convert to numpy arrays
    x = np.array(x_vals)
    y = np.array(y_vals)
    
    # create the handlebar points based on the mode
    # linear mode
    if model:  
        # compute the best fit line on the main data
        coef = np.polyfit(x, y, 1)
        slope = coef[0]
        intercept = coef[1]

        # choose two x-values for the handlebars
        top = np.percentile(x, 80)
        bottom = np.percentile(x, 20)

        # compute y-values from the best-fit line
        y_top = slope * top + intercept
        y_bottom = slope * bottom + intercept
        
        x_full = np.append(x, [top, bottom])
        y_full = np.append(y, [y_top, y_bottom])

    # logistic mode
    else: 
        # assume the provided y values are like probabilities,
        # but not perfect. fit a logistic regression on a binary version
        X = x.reshape(-1, 1)
        y_bin = (y >= 0.5).astype(int)
        mod = LogisticRegression()
        mod.fit(X, y_bin)

        # compute the best-fit logistic probabilities on main data
        y_fit = 1 / (1 + np.exp(-(mod.intercept_[0] + mod.coef_[0][0] * x)))
        top = np.percentile(x, 80)
        bottom = np.percentile(x, 20)
        y_top = 1 / (1 + np.exp(-(mod.intercept_[0] + mod.coef_[0][0] * top)))
        y_bottom = 1 / (1 + np.exp(-(mod.intercept_[0] + mod.coef_[0][0] * bottom)))
        x_full = np.append(x, [top, bottom])
        y_full = np.append(y_fit, [y_top, y_bottom])


    new_data = [np.array(x_full), np.array(y_full)]
    data = new_data
    if model:
        x_arr = new_data[0][:-2]
        y_arr = new_data[1][:-2]
        og_fit_line = np.polyval(np.polyfit(x_arr, y_arr, 1), x_arr)
    else:
        fit_line, _, _ = line_fit(new_data, False)
        og_fit_line = fit_line

        
    return {"x": x_full.tolist(), "y": y_full.tolist()}

# endpoint to provide the current data, model mode, and og_fit_line
@app.get("/get_data")
async def get_data():
    global data, model, og_fit_line
    if data is None:
        curr_mode = model if model is not None else True
        data = generate_data(curr_mode)
        if curr_mode:
            x_arr = data[0][:-2]
            y_arr = data[1][:-2]
            og_fit_line_local = np.polyval(np.polyfit(x_arr, y_arr, 1), x_arr)
        else:
            fit_line, _, _ = line_fit(data, False)
            og_fit_line_local = fit_line
        og_fit_line = og_fit_line_local

    if isinstance(og_fit_line, np.ndarray):
        og_fit_line_json = og_fit_line.tolist()
    else:
        og_fit_line_json = [og_fit_line[0].tolist(), og_fit_line[1].tolist()]
    return {
        "data": {
            "x": data[0].tolist(),
            "y": data[1].tolist()
        },
        "model": model,
        "og_fit_line": og_fit_line_json
    }

configure(app, InteractiveGraph)

# runs the program
def main():
   global model, data, og_fit_line
   if model is None:
       model = True  # begins in linear mode
   if data is None:
       data = generate_data(model)

    # linear mode
   if model: 
       x_arr = data[0][:-2]
       y_arr = data[1][:-2]
       og_fit_line = np.polyval(np.polyfit(x_arr, y_arr, 1), x_arr)

    # logistic mode
   else: 
       fit_line, resids, coef = line_fit(data, False)
       og_fit_line = fit_line
   uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
