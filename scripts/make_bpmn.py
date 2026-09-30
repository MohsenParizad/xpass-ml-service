"""Generate docs/xpass_process.bpmn: the ML lifecycle as a BPMN 2.0 process with 4 lanes.
Open the file in https://demo.bpmn.io or Camunda Modeler to view and edit it."""
from pathlib import Path
from xml.sax.saxutils import quoteattr

LANE_H, POOL_X, POOL_W, LANE_LABEL = 170, 100, 1640, 30
LANES = [("lane_data", "Data pipeline"), ("lane_model", "Modelling"),
         ("lane_serve", "CI/CD & Serving"), ("lane_mon", "Monitoring")]
TOP = 60

# id: (type, name, lane_index, x_center)
NODES = {
    "start_data": ("startEvent", "New tournament data available", 0, 200),
    "t_ingest": ("task", "Download StatsBomb events + 360 frames", 0, 340),
    "t_clean": ("task", "Clean + validate in-play passes", 0, 500),
    "t_features": ("task", "Build 13 features (shared with API)", 0, 660),
    "t_train": ("task", "Train + calibrate candidates (split by match)", 1, 820),
    "gw_gate": ("exclusiveGateway", "Evaluation gate passed?", 1, 970),
    "end_rejected": ("endEvent", "Model rejected, not deployed", 1, 970),
    "t_save": ("task", "Save versioned model + metadata", 1, 1110),
    "t_ci": ("task", "Run tests + build Docker image", 2, 1110),
    "t_deploy": ("task", "Deploy new Cloud Run revision", 2, 1270),
    "t_serve": ("task", "Serve /predict + log predictions", 2, 1430),
    "end_live": ("endEvent", "Service live", 2, 1590),
    "start_timer": ("startEvent", "Every Monday", 3, 200),
    "t_drift": ("task", "Drift check (PSI, Brier, coverage)", 3, 340),
    "gw_drift": ("exclusiveGateway", "Retraining recommended?", 3, 500),
    "end_ok": ("endEvent", "No action", 3, 500),
    "t_issue": ("task", "Open GitHub issue: retrain", 3, 740),
}
FLOWS = [
    ("start_data", "t_ingest", ""), ("t_ingest", "t_clean", ""), ("t_clean", "t_features", ""),
    ("t_features", "t_train", ""), ("t_train", "gw_gate", ""),
    ("gw_gate", "t_save", "yes"), ("gw_gate", "end_rejected", "no"),
    ("t_save", "t_ci", ""), ("t_ci", "t_deploy", ""), ("t_deploy", "t_serve", ""),
    ("t_serve", "end_live", ""),
    ("start_timer", "t_drift", ""), ("t_drift", "gw_drift", ""),
    ("gw_drift", "t_issue", "yes"), ("gw_drift", "end_ok", "no"),
    ("t_issue", "start_data", "triggers retraining"),
]
SIZE = {"task": (120, 80), "startEvent": (36, 36), "endEvent": (36, 36),
        "exclusiveGateway": (50, 50)}


def lane_y(i):
    return TOP + i * LANE_H


def bounds(nid):
    t, _, lane, cx = NODES[nid]
    w, h = SIZE[t]
    cy = lane_y(lane) + LANE_H / 2
    if nid in ("end_rejected", "end_ok"):  # place below the gateway
        cy += 68
    return cx - w / 2, cy - h / 2, w, h


def waypoints(src, tgt):
    sx, sy, sw, sh = bounds(src)
    tx, ty, tw, th = bounds(tgt)
    if src == "t_issue":  # loop back along the left margin to the data start event
        ytop = lane_y(3) + 18
        return [(sx + sw / 2, sy), (sx + sw / 2, ytop), (168, ytop), (168, ty + th / 2),
                (tx, ty + th / 2)]
    if tgt in ("end_rejected", "end_ok"):
        return [(sx + sw / 2, sy + sh), (tx + tw / 2, ty)]
    a, b = (sx + sw, sy + sh / 2), (tx, ty + th / 2)
    if abs(a[1] - b[1]) < 1:
        return [a, b]
    if tx + tw / 2 <= sx + sw:  # target directly below: go down from the bottom
        return [(sx + sw / 2, sy + sh), (tx + tw / 2, ty)]
    mid = (a[0] + b[0]) / 2
    return [a, (mid, a[1]), (mid, b[1]), b]


def xml():
    proc, shapes, edges = [], [], []
    for i, (lid, lname) in enumerate(LANES):
        refs = "".join(f"<bpmn:flowNodeRef>{n}</bpmn:flowNodeRef>" for n, v in NODES.items() if v[2] == i)
        proc.append(f'<bpmn:lane id="{lid}" name={quoteattr(lname)}>{refs}</bpmn:lane>')
        shapes.append(f'<bpmndi:BPMNShape id="{lid}_di" bpmnElement="{lid}" isHorizontal="true">'
                      f'<dc:Bounds x="{POOL_X + LANE_LABEL}" y="{lane_y(i)}" width="{POOL_W - LANE_LABEL}" height="{LANE_H}"/></bpmndi:BPMNShape>')
    proc = [f'<bpmn:laneSet id="laneset">{"".join(proc)}</bpmn:laneSet>']
    incoming = {n: [] for n in NODES}
    outgoing = {n: [] for n in NODES}
    for k, (s, t, _) in enumerate(FLOWS):
        outgoing[s].append(f"f{k}")
        incoming[t].append(f"f{k}")
    for nid, (typ, name, _, _) in NODES.items():
        inner = "".join(f"<bpmn:incoming>{f}</bpmn:incoming>" for f in incoming[nid])
        inner += "".join(f"<bpmn:outgoing>{f}</bpmn:outgoing>" for f in outgoing[nid])
        if nid == "start_timer":
            inner += '<bpmn:timerEventDefinition id="timer_def"><bpmn:timeCycle xsi:type="bpmn:tFormalExpression">R/P1W</bpmn:timeCycle></bpmn:timerEventDefinition>'
        proc.append(f'<bpmn:{typ} id="{nid}" name={quoteattr(name)}>{inner}</bpmn:{typ}>')
        x, y, w, h = bounds(nid)
        extra = ' isMarkerVisible="true"' if typ == "exclusiveGateway" else ""
        label = ""
        if typ != "task":
            label = f'<bpmndi:BPMNLabel><dc:Bounds x="{x - 30}" y="{y + h + 4}" width="{w + 60}" height="27"/></bpmndi:BPMNLabel>'
            if typ == "exclusiveGateway":
                label = f'<bpmndi:BPMNLabel><dc:Bounds x="{x - 20}" y="{y - 32}" width="{w + 40}" height="27"/></bpmndi:BPMNLabel>'
        shapes.append(f'<bpmndi:BPMNShape id="{nid}_di" bpmnElement="{nid}"{extra}>'
                      f'<dc:Bounds x="{x}" y="{y}" width="{w}" height="{h}"/>{label}</bpmndi:BPMNShape>')
    for k, (s, t, name) in enumerate(FLOWS):
        nm = f" name={quoteattr(name)}" if name else ""
        proc.append(f'<bpmn:sequenceFlow id="f{k}"{nm} sourceRef="{s}" targetRef="{t}"/>')
        wps = "".join(f'<di:waypoint x="{x:.0f}" y="{y:.0f}"/>' for x, y in waypoints(s, t))
        lbl = ""
        if name:
            wp = waypoints(s, t)
            if name == "yes":
                lx, ly = (wp[0][0] + wp[1][0]) / 2 - 60, wp[0][1] - 18
            elif name == "no":
                lx, ly = wp[0][0] + 16 - 60, (wp[0][1] + wp[1][1]) / 2 - 7
            else:  # loop label on the horizontal segment at the top of the monitoring lane
                lx, ly = (wp[1][0] + wp[2][0]) / 2 - 60, wp[1][1] - 16
            lbl = f'<bpmndi:BPMNLabel><dc:Bounds x="{lx:.0f}" y="{ly:.0f}" width="120" height="14"/></bpmndi:BPMNLabel>'
        edges.append(f'<bpmndi:BPMNEdge id="f{k}_di" bpmnElement="f{k}">{wps}{lbl}</bpmndi:BPMNEdge>')
    total_h = LANE_H * len(LANES)
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL"
  xmlns:bpmndi="http://www.omg.org/spec/BPMN/20100524/DI" xmlns:dc="http://www.omg.org/spec/DD/20100524/DC"
  xmlns:di="http://www.omg.org/spec/DD/20100524/DI" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  id="xpass_defs" targetNamespace="http://bpmn.io/schema/bpmn">
  <bpmn:collaboration id="collab">
    <bpmn:participant id="pool" name="xPass ML service" processRef="xpass_process"/>
  </bpmn:collaboration>
  <bpmn:process id="xpass_process" name="xPass ML lifecycle" isExecutable="false">
    {"".join(proc)}
  </bpmn:process>
  <bpmndi:BPMNDiagram id="diagram">
    <bpmndi:BPMNPlane id="plane" bpmnElement="collab">
      <bpmndi:BPMNShape id="pool_di" bpmnElement="pool" isHorizontal="true">
        <dc:Bounds x="{POOL_X}" y="{TOP}" width="{POOL_W}" height="{total_h}"/>
      </bpmndi:BPMNShape>
      {"".join(shapes)}
      {"".join(edges)}
    </bpmndi:BPMNPlane>
  </bpmndi:BPMNDiagram>
</bpmn:definitions>
'''


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "docs" / "xpass_process.bpmn"
    out.write_text(xml())
    print(out)
