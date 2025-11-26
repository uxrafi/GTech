// Q2.js
// Author: Umar Rafi (urafi3)
// Georgia Tech - CSE 6242

const width = 1000;
const height = 1000;

const svg = d3.select("body")
    .append("svg")
    .attr("width", width)
    .attr("height", height);

// Add GT username at top right corner
svg.append("text")
    .attr("id", "credit")
    .attr("x", width - 100)
    .attr("y", 30)
    .text("urafi3")
    .attr("font-size", "14px")
    .attr("fill", "black");

// Load data
d3.csv("board_games.csv").then(function(data) {
    // Build links and nodes
    const links = data.map(d => ({
        source: d.source,
        target: d.target,
        value: +d.value
    }));

    // Build node list
    const nodes = {};
    links.forEach(l => {
        nodes[l.source] = nodes[l.source] || { id: l.source, weight: 0 };
        nodes[l.target] = nodes[l.target] || { id: l.target, weight: 0 };
        nodes[l.source].weight += 1;
        nodes[l.target].weight += 1;
    });

    const nodeList = Object.values(nodes);

    // Scales
    const sizeScale = d3.scaleLinear()
        .domain(d3.extent(nodeList, d => d.weight))
        .range([8, 30]);

    const colorScale = d3.scaleSequential(d3.interpolateBlues)
        .domain(d3.extent(nodeList, d => d.weight));

    // Create links
    const link = svg.selectAll(".link")
        .data(links)
        .enter()
        .append("line")
        .attr("class", "link")
        .style("stroke", d => d.value > 0 ? "red" : "gray")
        .style("stroke-width", 1.5)
        .style("stroke-dasharray", d => d.value > 0 ? "5,5" : "0");

    // Create nodes (group for circle + text)
    const node = svg.selectAll(".node")
        .data(nodeList)
        .enter()
        .append("g")
        .attr("class", "node")
        .attr("id", d => d.id)
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended));

    node.append("circle")
        .attr("r", d => sizeScale(d.weight))
        .attr("fill", d => colorScale(d.weight))
        .attr("stroke", "black")
        .attr("stroke-width", 1.5);

    node.append("text")
        .text(d => d.id)
        .attr("x", 6)
        .attr("y", 3)
        .style("font-size", "10px");

    // Simulation
    const simulation = d3.forceSimulation(nodeList)
        .force("link", d3.forceLink(links).id(d => d.id).distance(150))
        .force("charge", d3.forceManyBody().strength(-400))
        .force("center", d3.forceCenter(width / 2, height / 2));

    simulation.on("tick", () => {
        link
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

        node.attr("transform", d => `translate(${d.x},${d.y})`);
    });

    // Drag functions
    function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
        d.fixed = true;
        d3.select(this).select("circle").attr("fill", "yellow"); // color change when pinned
    }

    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = d.x;
        d.fy = d.y;
        d.fixed = true; // mark pinned
    }

    // Double-click to unpin and restore color
    node.on("dblclick", function(event, d) {
        d.fx = null;
        d.fy = null;
        d.fixed = false;
        d3.select(this).select("circle").attr("fill", colorScale(d.weight));
    });
});
