// Shared page library for the fuzzy-LM dashboards. Defines window.FL; page scripts call it.
(function () {
  const fmtInt = d3.format(',');
  const f2 = d3.format('.2f');
  const f3 = d3.format('.3f');
  const f4 = d3.format('.4f');

  function get(id) {
    const s = dash.data(id);
    return { ok: s.status === 'ok', status: s.status, rows: s.status === 'ok' ? s.data : [] };
  }
  const num = (x) => (x === '' || x == null ? NaN : Number(x));
  function stats(vals) {
    const v = vals.filter((x) => Number.isFinite(x));
    const n = v.length;
    const mean = n ? d3.mean(v) : NaN;
    const std = n > 1 ? d3.deviation(v) : NaN;
    return { n, mean, std };
  }

  function fail(el) {
    el.textContent = '';
    el.classList.remove('dash-skeleton');
    const d = document.createElement('div');
    d.className = 'fl-fail';
    d.textContent = 'Failed to load';
    el.append(d);
  }
  function pending(el, st) {
    if (st === 'error') { fail(el); return true; }
    if (st !== 'ok') { el.textContent = ''; el.classList.add('dash-skeleton'); return true; }
    el.classList.remove('dash-skeleton');
    return false;
  }

  // Fill every single-value mark under root from its dataset or calc.
  function fill(root) {
    for (const el of root.querySelectorAll('[data-source][data-field]')) {
      if (el.tagName === 'TH') continue;
      const st = dash.data(el.dataset.source);
      if (st.status !== 'ok') { el.textContent = '—'; if (st.status === 'error') el.classList.remove('dash-skeleton'); continue; }
      let row;
      if (el.dataset.row != null) row = st.data[Number(el.dataset.row)];
      else if (el.dataset.where) {
        const conds = [...new URLSearchParams(el.dataset.where)];
        row = st.data.find((r) => conds.every(([k, v]) => String(r[k]) === v));
      }
      el.classList.remove('dash-skeleton');
      el.textContent = row ? String(row[el.dataset.field]) : '—';
    }
  }

  // Tooltip helper: a small popover inside a position:relative box.
  function tipFor(el) {
    const tip = d3.select(el).append('div').attr('class', 'fl-tip').style('display', 'none');
    return {
      show(html, x, y, W) {
        tip.style('display', null).html('');
        for (const [cls, text] of html) tip.append('div').attr('class', cls).text(text);
        const box = el.getBoundingClientRect();
        const tw = tip.node().offsetWidth;
        const sx = (x / W) * box.width;
        tip.style('left', Math.min(Math.max(0, sx + 12), box.width - tw) + 'px').style('top', Math.max(0, y - 44) + 'px');
      },
      hide() { tip.style('display', 'none'); },
    };
  }

  // Multi-series line chart. o: {series:[{name,color,width,points:[{x,y,lo,hi,label}]}], xLog,yLog,height,xLabel,yFmt,xFmt,xTickValues,refs,yTicks,yDomain}
  function lines(el, o) {
    el.textContent = '';
    const W = Math.max(280, el.clientWidth);
    const H = o.height || 300;
    const svg = d3.select(el).append('svg').attr('width', '100%').attr('viewBox', `0 0 ${W} ${H}`);
    const all = o.series.flatMap((s) => s.points);
    if (!all.length) return;
    const ys = all.flatMap((p) => [p.lo ?? p.y, p.hi ?? p.y]).concat((o.refs || []).map((r) => r.y));
    const yExt = o.yDomain || d3.extent(ys);
    const pad = (yExt[1] - yExt[0]) * 0.06 || 0.05;
    const y = (o.yLog ? d3.scaleLog() : d3.scaleLinear()).domain(o.yLog || o.yDomain ? yExt : [yExt[0] - pad, yExt[1] + pad]).nice();
    const yTicks = y.ticks(o.yTicks || 6);
    const probe = svg.append('text').attr('visibility', 'hidden');
    const left = d3.max(yTicks, (t) => probe.text(o.yFmt(t)).node().getComputedTextLength()) + 14;
    probe.remove();
    const m = { top: 8, right: 16, bottom: 36, left };
    const x = (o.xLog ? d3.scaleLog() : d3.scaleLinear()).domain(d3.extent(all, (p) => p.x)).range([m.left, W - m.right]);
    y.range([H - m.bottom, m.top]);
    svg.append('g').selectAll('line').data(yTicks).join('line')
      .attr('x1', m.left).attr('x2', W - m.right).attr('y1', (t) => y(t)).attr('y2', (t) => y(t))
      .style('stroke', 'var(--cds-chart-grid)').attr('stroke-width', 1);
    svg.append('g').selectAll('text').data(yTicks).join('text')
      .attr('x', m.left - 8).attr('y', (t) => y(t)).attr('dy', '0.32em').attr('text-anchor', 'end').text(o.yFmt);
    const xTicks = o.xTickValues || x.ticks(Math.max(2, Math.floor(W / 110)));
    const xg = svg.append('g').attr('transform', `translate(0,${H - m.bottom})`);
    xg.append('line').attr('x1', m.left).attr('x2', W - m.right).style('stroke', 'var(--cds-chart-axis)');
    xg.selectAll('text').data(xTicks).join('text').attr('x', (t) => x(t)).attr('y', 18).attr('text-anchor', 'middle').text(o.xFmt);
    svg.append('text').attr('x', W - m.right).attr('y', H - 2).attr('text-anchor', 'end').text(o.xLabel);
    for (const r of o.refs || []) {
      svg.append('line').attr('x1', m.left).attr('x2', W - m.right).attr('y1', y(r.y)).attr('y2', y(r.y))
        .style('stroke', 'var(--cds-chart-reference)').attr('stroke-dasharray', '4 4');
      svg.append('text').attr('x', m.left + 6).attr('y', y(r.y) - 5).text(r.label);
    }
    for (const s of o.series) {
      const band = s.points.filter((p) => p.lo != null && p.hi != null);
      if (band.length > 1) {
        svg.append('path').datum(band).style('fill', s.color).attr('fill-opacity', 0.12).attr('stroke', 'none')
          .attr('d', d3.area().x((p) => x(p.x)).y0((p) => y(p.lo)).y1((p) => y(p.hi)));
      }
      svg.append('path').datum(s.points).attr('fill', 'none').style('stroke', s.color)
        .attr('stroke-width', s.width || 2).attr('stroke-linecap', 'round').attr('stroke-linejoin', 'round')
        .attr('d', d3.line().x((p) => x(p.x)).y((p) => y(p.y)));
      svg.append('g').selectAll('circle').data(s.points).join('circle')
        .attr('cx', (p) => x(p.x)).attr('cy', (p) => y(p.y)).attr('r', s.width > 2 ? 4 : 3).style('fill', s.color);
    }
    const tip = tipFor(el);
    const dot = svg.append('circle').attr('r', 6).attr('fill', 'none').attr('stroke-width', 2).style('display', 'none');
    const flat = o.series.flatMap((s) => s.points.map((p) => ({ ...p, s })));
    const del = d3.Delaunay.from(flat, (p) => x(p.x), (p) => y(p.y));
    svg.append('rect').attr('x', m.left).attr('y', m.top).attr('width', W - m.left - m.right).attr('height', H - m.top - m.bottom)
      .attr('fill', 'transparent')
      .on('pointermove', (ev) => {
        const [px, py] = d3.pointer(ev);
        const p = flat[del.find(px, py)];
        dot.style('display', null).attr('cx', x(p.x)).attr('cy', y(p.y)).style('stroke', p.s.color);
        tip.show([['fl-tip-head', p.s.name], ['', p.label]], x(p.x), (y(p.y) / H) * el.getBoundingClientRect().height, W);
      })
      .on('pointerleave', () => { tip.hide(); dot.style('display', 'none'); });
  }

  // Horizontal bars. items: [{label, value, err, color, note, group}] drawn in the given order.
  // o: {xDomain, xFmt, valueFmt, xLabel, rowH, refs:[{x,label}], accent: label|fn}
  function bars(el, items, o) {
    el.textContent = '';
    const W = Math.max(300, el.clientWidth);
    const rowH = o.rowH || 28;
    const gapG = 20;
    const probe = d3.select(el).append('svg').attr('width', 0).attr('height', 0);
    const t = probe.append('text');
    const left = Math.min(W * 0.46, d3.max(items, (d) => t.text(d.label).node().getComputedTextLength()) + 14);
    const notes = d3.max(items, (d) => t.text(d.note || '').node().getComputedTextLength()) || 0;
    probe.remove();
    const groups = [...new Set(items.map((d) => d.group || ''))];
    const H = items.length * rowH + (groups.length > 1 ? groups.length * gapG : 0) + 34;
    const svg = d3.select(el).append('svg').attr('width', '100%').attr('viewBox', `0 0 ${W} ${H}`);
    const right = notes + 16;
    const lo = o.xDomain ? o.xDomain[0] : 0;
    const hi = o.xDomain ? o.xDomain[1] : d3.max(items, (d) => d.value + (d.err || 0));
    const x = d3.scaleLinear().domain([lo, hi]).nice().range([left, W - right - 8]);
    const ticks = x.ticks(5);
    svg.append('g').selectAll('line').data(ticks).join('line').attr('x1', (v) => x(v)).attr('x2', (v) => x(v))
      .attr('y1', 0).attr('y2', H - 24).style('stroke', 'var(--cds-chart-grid)');
    svg.append('g').selectAll('text').data(ticks).join('text').attr('x', (v) => x(v)).attr('y', H - 8).attr('text-anchor', 'middle').text(o.xFmt || f2);
    if (o.xLabel) svg.append('text').attr('x', W - right - 8).attr('y', H - 8).attr('text-anchor', 'end').attr('dx', '0').text('');
    for (const r of o.refs || []) {
      svg.append('line').attr('x1', x(r.x)).attr('x2', x(r.x)).attr('y1', 0).attr('y2', H - 24)
        .style('stroke', 'var(--cds-chart-reference)').attr('stroke-dasharray', '4 4');
    }
    const tip = tipFor(el);
    let yy = 0;
    let lastGroup = null;
    items.forEach((d, i) => {
      if (groups.length > 1 && d.group !== lastGroup) {
        svg.append('text').attr('x', 0).attr('y', yy + 14).style('fill', 'var(--color-fg)').text(d.group || '');
        yy += gapG;
        lastGroup = d.group;
      }
      const cy = yy + rowH / 2;
      const c = d.color || 'var(--cds-chart-muted)';
      svg.append('text').attr('x', left - 10).attr('y', cy).attr('dy', '0.32em').attr('text-anchor', 'end').text(d.label);
      const b0 = o.base != null ? o.base : lo;  // signed bars grow from the base value
      const bx = Math.min(x(b0), x(d.value));
      svg.append('rect').attr('x', bx).attr('y', cy - 7).attr('width', Math.abs(x(d.value) - x(b0))).attr('height', 14).attr('rx', 4).style('fill', c);
      if (d.err) {
        svg.append('line').attr('x1', x(d.value - d.err)).attr('x2', x(d.value + d.err)).attr('y1', cy).attr('y2', cy)
          .style('stroke', 'var(--color-fg)').attr('stroke-width', 1.5);
      }
      const tx = d.value < (o.base != null ? o.base : lo) ? x(d.value - (d.err || 0)) - 8 : x(d.value + (d.err || 0)) + 8;
      svg.append('text').attr('x', tx).attr('y', cy).attr('dy', '0.32em').attr('text-anchor', d.value < (o.base != null ? o.base : lo) ? 'end' : 'start').text((o.valueFmt || f3)(d.value));
      if (d.note) svg.append('text').attr('x', W - 4).attr('y', cy).attr('dy', '0.32em').attr('text-anchor', 'end').text(d.note);
      svg.append('rect').attr('x', 0).attr('y', yy).attr('width', W).attr('height', rowH).attr('fill', 'transparent')
        .on('pointermove', () => tip.show([['fl-tip-head', d.label], ['', d.tip || `${(o.valueFmt || f3)(d.value)}${d.err ? ' ± ' + f3(d.err) : ''}${d.note ? ' · ' + d.note : ''}`]], x(d.value), cy + 18, W))
        .on('pointerleave', () => tip.hide());
      yy += rowH;
    });
  }

  // Fill a <tbody> from rows with the given column keys; formatters: {key: fn(row)->string}
  function table(tbody, rows, cols, fmt) {
    tbody.textContent = '';
    for (const r of rows) {
      const tr = document.createElement('tr');
      if (r._key != null) tr.dataset.key = r._key;
      if (r._focus) tr.className = 'fl-focus';
      for (const c of cols) {
        const td = document.createElement('td');
        td.textContent = fmt && fmt[c] ? fmt[c](r) : r[c];
        tr.append(td);
      }
      tbody.append(tr);
    }
  }

  // Pages: templates cloned in one at a time; the dashboard link follows the page.
  function pages(spec) {
    const main = document.getElementById('fl-main');
    const tabs = [...document.querySelectorAll('.fl-tab')];
    let current = null;
    function show(name, push) {
      if (!spec[name]) name = tabs[0].dataset.page;
      current = name;
      main.replaceChildren(document.getElementById('tpl-' + name).content.cloneNode(true));
      for (const t of tabs) t.setAttribute('aria-selected', String(t.dataset.page === name));
      if (push) dash.setLink('tab-' + name);
      draw();
    }
    function draw() {
      if (!current) return;
      fill(main);
      spec[current](main);
    }
    for (const t of tabs) t.addEventListener('click', () => show(t.dataset.page, true));
    const named = (location.hash || '').replace('#tab-', '').replace('#', '');
    show(spec[named] ? named : tabs[0].dataset.page, false);
    dash.onData(draw);
  }

  window.FL = { fmtInt, f2, f3, f4, get, num, stats, fail, pending, fill, lines, bars, table, pages, tipFor };
})();
