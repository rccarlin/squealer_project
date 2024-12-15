from fastapi import FastAPI
from reactpy import component, html
from reactpy.backend.fastapi import configure
import uvicorn



# Define the main ReactPy component with D3 integration
@component
def InteractivePlot():
    points = [{"x": 1, "y": 2}, {"x": 2, "y": 4}, {"x": 3, "y": 6}, {"x": 4, "y": 8}]
    best_fit_line = [{"x": 1, "y": 1.5}, {"x": 4, "y": 7.5}]
    handle_bar_points = [{"x": 1, "y": 1.5}, {"x": 4, "y": 7.5}]

    return html.div(
        [
            html.script(src="https://d3js.org/d3.v7.min.js"),
            html.div(
                id="plot-container",
                style={"width": "800px", "height": "600px", "border": "1px solid black"},
            ),
            html.script(
                f"""
                document.addEventListener("DOMContentLoaded", function() {{
                    const points = {points};
                    const bestFitLine = {best_fit_line};
                    const handleBarPoints = {handle_bar_points};

                    const width = 800;
                    const height = 600;
                    const margin = {{ top: 20, right: 30, bottom: 50, left: 50 }};

                    const svg = d3.select("#plot-container")
                        .append("svg")
                        .attr("width", width)
                        .attr("height", height);

                    const xScale = d3.scaleLinear().domain([0, d3.max(points, d => d.x)]).range([margin.left, width - margin.right]);
                    const yScale = d3.scaleLinear().domain([0, d3.max(points, d => d.y)]).range([height - margin.bottom, margin.top]);

                    svg.selectAll("circle")
                        .data(points)
                        .enter()
                        .append("circle")
                        .attr("cx", d => xScale(d.x))
                        .attr("cy", d => yScale(d.y))
                        .attr("r", 5)
                        .style("fill", "blue");

                    svg.append("line")
                        .attr("x1", xScale(bestFitLine[0].x))
                        .attr("y1", yScale(bestFitLine[0].y))
                        .attr("x2", xScale(bestFitLine[1].x))
                        .attr("y2", yScale(bestFitLine[1].y))
                        .attr("stroke", "red")
                        .attr("stroke-width", 2);

                    svg.selectAll(".handle-bar")
                        .data(handleBarPoints)
                        .enter()
                        .append("circle")
                        .attr("class", "handle-bar")
                        .attr("cx", d => xScale(d.x))
                        .attr("cy", d => yScale(d.y))
                        .attr("r", 8)
                        .style("fill", "green")
                        .style("cursor", "pointer")
                        .call(d3.drag()
                            .on("drag", function(event, d) {{
                                const newX = event.x;
                                const newY = event.y;
                                d.x = xScale.invert(newX);
                                d.y = yScale.invert(newY);

                                d3.select(this)
                                    .attr("cx", newX)
                                    .attr("cy", newY);

                                // Update the line
                                svg.select("line")
                                    .attr("x1", xScale(handleBarPoints[0].x))
                                    .attr("y1", yScale(handleBarPoints[0].y))
                                    .attr("x2", xScale(handleBarPoints[1].x))
                                    .attr("y2", yScale(handleBarPoints[1].y));
                            }}));
                }});
                """
            ),
        ]
    )

# Create the FastAPI app
app = FastAPI()

# Integrate the ReactPy component into FastAPI
configure(app, InteractivePlot)
uvicorn.run(app, host="127.0.0.1", port=8000)

# To run the app:
# Save this script as app.py and run `uvicorn app:app --reload`
# Access the app at http://127.0.0.1:8000
