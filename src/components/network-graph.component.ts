
import { Component, ElementRef, ViewChild, effect, input, OnDestroy, AfterViewInit } from '@angular/core';
import * as d3 from 'd3';

export interface GraphNode {
  id: string;
  role: 'origin' | 'destination' | 'hub';
  risk: number;        // 0-1
  predictedDelay: number;
  congestion: number;
  precipSeverity: number;
  condition: string;
  wind: number;
  visibility: number;
  lat: number | null;
  lon: number | null;
}

export interface GraphEdge {
  source: string;
  target: string;
  type: 'flight' | 'inbound' | 'network';
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

@Component({
  selector: 'app-network-graph',
  standalone: true,
  template: `
    <div class="w-full h-full min-h-[300px] relative bg-slate-900/50 rounded-xl overflow-hidden border border-slate-700/50">
      <div #graphContainer class="w-full h-full absolute inset-0"></div>
      <div class="absolute top-3 left-3 text-[10px] font-mono text-slate-500 bg-slate-900/90 px-2 py-1.5 rounded border border-slate-700/50 space-y-1">
        <div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-cyan-400"></span> Origin</div>
        <div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-emerald-400"></span> Destination</div>
        <div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-slate-500"></span> Hub (low risk)</div>
        <div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-rose-500"></span> Hub (high risk)</div>
        <div class="flex items-center gap-1.5"><span class="w-3 h-[2px] bg-cyan-400"></span> Flight path</div>
        <div class="flex items-center gap-1.5"><span class="w-3 h-[2px] bg-slate-600"></span> Network edge</div>
      </div>
    </div>
  `
})
export class NetworkGraphComponent implements OnDestroy, AfterViewInit {
  graphData = input.required<GraphData | null>();
  @ViewChild('graphContainer') graphContainer!: ElementRef;

  private resizeObserver: ResizeObserver | null = null;

  constructor() {
    effect(() => {
      const data = this.graphData();
      if (this.graphContainer && data) {
        this.renderGraph(data);
      }
    });
  }

  ngAfterViewInit() {
    const data = this.graphData();
    if (data) {
      this.renderGraph(data);
    }
    // Re-render on resize
    this.resizeObserver = new ResizeObserver(() => {
      const d = this.graphData();
      if (d) this.renderGraph(d);
    });
    this.resizeObserver.observe(this.graphContainer.nativeElement);
  }

  ngOnDestroy() {
    if (this.resizeObserver) this.resizeObserver.disconnect();
  }

  private renderGraph(graphData: GraphData) {
    if (!this.graphContainer) return;
    const el = this.graphContainer.nativeElement;

    d3.select(el).selectAll('*').remove();

    const width = el.clientWidth || 400;
    const height = el.clientHeight || 300;
    const pad = 30;

    const svg = d3.select(el)
      .append('svg')
      .attr('width', '100%')
      .attr('height', '100%')
      .attr('viewBox', `0 0 ${width} ${height}`);

    // Defs for arrowheads and gradients
    const defs = svg.append('defs');

    // Glow filter for origin/destination
    const glow = defs.append('filter').attr('id', 'glow');
    glow.append('feGaussianBlur').attr('stdDeviation', '3').attr('result', 'blur');
    glow.append('feMerge')
      .selectAll('feMergeNode')
      .data(['blur', 'SourceGraphic'])
      .join('feMergeNode')
      .attr('in', (d: string) => d);

    const { nodes, edges } = graphData;
    if (!nodes.length) return;

    // Position nodes using lat/lon if available, otherwise force layout
    const hasGeo = nodes.filter(n => n.lat != null && n.lon != null).length > nodes.length * 0.5;

    type PosNode = GraphNode & { x: number; y: number };
    let posNodes: PosNode[];

    if (hasGeo) {
      // Project lat/lon onto the SVG with a simple Mercator-like mapping
      const lats = nodes.filter(n => n.lat != null).map(n => n.lat!);
      const lons = nodes.filter(n => n.lon != null).map(n => n.lon!);
      const minLat = Math.min(...lats), maxLat = Math.max(...lats);
      const minLon = Math.min(...lons), maxLon = Math.max(...lons);
      const latRange = Math.max(maxLat - minLat, 5);
      const lonRange = Math.max(maxLon - minLon, 5);

      posNodes = nodes.map(n => {
        const lat = n.lat ?? (minLat + latRange / 2);
        const lon = n.lon ?? (minLon + lonRange / 2);
        return {
          ...n,
          x: pad + ((lon - minLon) / lonRange) * (width - 2 * pad),
          y: pad + ((maxLat - lat) / latRange) * (height - 2 * pad), // invert Y
        };
      });
    } else {
      // Fallback: circular layout
      const cx = width / 2, cy = height / 2;
      const r = Math.min(width, height) / 2 - pad - 20;
      // Put origin/dest in prominent positions
      const originNode = nodes.find(n => n.role === 'origin');
      const destNode = nodes.find(n => n.role === 'destination');
      const hubs = nodes.filter(n => n.role === 'hub');

      posNodes = [];
      if (originNode) posNodes.push({ ...originNode, x: cx - r * 0.6, y: cy });
      if (destNode) posNodes.push({ ...destNode, x: cx + r * 0.6, y: cy });
      hubs.forEach((n, i) => {
        const angle = (i / hubs.length) * Math.PI * 2 - Math.PI / 2;
        posNodes.push({ ...n, x: cx + Math.cos(angle) * r, y: cy + Math.sin(angle) * r });
      });
    }

    const nodeMap = new Map(posNodes.map(n => [n.id, n]));

    // --- Draw edges ---
    const edgeGroup = svg.append('g');

    edges.forEach(e => {
      const src = nodeMap.get(e.source);
      const dst = nodeMap.get(e.target);
      if (!src || !dst) return;

      const isFlight = e.type === 'flight';
      const isInbound = e.type === 'inbound';

      edgeGroup.append('line')
        .attr('x1', src.x).attr('y1', src.y)
        .attr('x2', dst.x).attr('y2', dst.y)
        .attr('stroke', isFlight ? '#22d3ee' : isInbound ? '#475569' : '#1e293b')
        .attr('stroke-width', isFlight ? 2.5 : 0.8)
        .attr('stroke-opacity', isFlight ? 0.9 : isInbound ? 0.4 : 0.15)
        .attr('stroke-dasharray', isFlight ? 'none' : '3,3');
    });

    // Animated flight path
    const flightEdge = edges.find(e => e.type === 'flight');
    if (flightEdge) {
      const src = nodeMap.get(flightEdge.source);
      const dst = nodeMap.get(flightEdge.target);
      if (src && dst) {
        // Animated dot along flight path
        const dot = svg.append('circle')
          .attr('r', 3)
          .attr('fill', '#22d3ee')
          .attr('filter', 'url(#glow)');

        function animateDot() {
          dot.attr('cx', src!.x).attr('cy', src!.y)
            .transition()
            .duration(3000)
            .ease(d3.easeLinear)
            .attr('cx', dst!.x)
            .attr('cy', dst!.y)
            .on('end', animateDot);
        }
        animateDot();
      }
    }

    // --- Draw nodes ---
    const nodeGroup = svg.append('g');

    posNodes.forEach(n => {
      const g = nodeGroup.append('g')
        .attr('transform', `translate(${n.x}, ${n.y})`);

      // Node size
      const isEndpoint = n.role === 'origin' || n.role === 'destination';
      const radius = isEndpoint ? 10 : 5 + n.risk * 4;

      // Node color
      let fill: string;
      if (n.role === 'origin') fill = '#22d3ee';       // cyan
      else if (n.role === 'destination') fill = '#34d399'; // emerald
      else {
        // Risk-based gradient: slate → amber → rose
        if (n.risk < 0.3) fill = '#64748b';
        else if (n.risk < 0.6) fill = '#f59e0b';
        else fill = '#ef4444';
      }

      // Glow ring for endpoints
      if (isEndpoint) {
        g.append('circle')
          .attr('r', radius + 4)
          .attr('fill', 'none')
          .attr('stroke', fill)
          .attr('stroke-width', 1.5)
          .attr('opacity', 0.4);
      }

      // Pulse ring for high-risk nodes
      if (n.risk > 0.6) {
        const pulse = g.append('circle')
          .attr('r', radius)
          .attr('fill', 'none')
          .attr('stroke', '#ef4444')
          .attr('stroke-width', 1.5);

        pulse.append('animate')
          .attr('attributeName', 'r')
          .attr('from', radius)
          .attr('to', radius + 12)
          .attr('dur', '2s')
          .attr('repeatCount', 'indefinite');
        pulse.append('animate')
          .attr('attributeName', 'opacity')
          .attr('from', 0.8)
          .attr('to', 0)
          .attr('dur', '2s')
          .attr('repeatCount', 'indefinite');
      }

      // Main circle
      g.append('circle')
        .attr('r', radius)
        .attr('fill', fill)
        .attr('stroke', '#0f172a')
        .attr('stroke-width', 1.5)
        .attr('filter', isEndpoint ? 'url(#glow)' : 'none');

      // Label
      const labelOffset = radius + 6;
      g.append('text')
        .text(n.id)
        .attr('x', 0)
        .attr('y', -labelOffset)
        .attr('text-anchor', 'middle')
        .attr('fill', isEndpoint ? '#e2e8f0' : '#94a3b8')
        .attr('font-size', isEndpoint ? '11px' : '9px')
        .attr('font-weight', isEndpoint ? 'bold' : 'normal')
        .attr('font-family', 'monospace');

      // Tooltip-style detail for origin/destination
      if (isEndpoint) {
        const detail = n.role === 'origin'
          ? `${n.condition} · ${n.wind}kt`
          : `Delay: ${n.predictedDelay.toFixed(0)}m`;
        g.append('text')
          .text(detail)
          .attr('x', 0)
          .attr('y', labelOffset + 10)
          .attr('text-anchor', 'middle')
          .attr('fill', '#64748b')
          .attr('font-size', '8px')
          .attr('font-family', 'monospace');
      }
    });
  }
}
