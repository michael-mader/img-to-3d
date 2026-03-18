import streamlit as st
import cv2
import numpy as np
import cadquery as cq
import tempfile
import os
import trimesh
import plotly.graph_objects as go
import time

class WindowProcessor:
    def __init__(self, height, target_width, do_deskew, min_area, smoothness):
        self.height = height
        self.target_width = target_width
        self.do_deskew = do_deskew
        self.min_area = min_area
        self.smoothness = smoothness

    def _apply_deskew(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 50, 200)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        img_area = img.shape[0] * img.shape[1]

        for c in contours:
            if cv2.contourArea(c) > 0.90 * img_area: continue
            approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype("float32")
                s = pts.sum(axis=1)
                diff = np.diff(pts, axis=1)
                rect = np.zeros((4, 2), dtype="float32")
                rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
                rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
                width = max(np.linalg.norm(rect[0]-rect[1]), np.linalg.norm(rect[2]-rect[3]))
                height = max(np.linalg.norm(rect[0]-rect[3]), np.linalg.norm(rect[1]-rect[2]))
                dst = np.array([[0,0], [width-1,0], [width-1,height-1], [0,height-1]], dtype="float32")
                M = cv2.getPerspectiveTransform(rect, dst)
                return cv2.warpPerspective(img, M, (int(width), int(height)))
        return img

    def process(self, image_bytes, progress_bar, status_text):
        t0 = time.time()
        
        status_text.text("⏱️ Step 1/5: Decoding image...")
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        
        t1 = time.time()
        status_text.text("⏱️ Step 2/5: Applying filters...")
        work_img = self._apply_deskew(img) if self.do_deskew else img
        gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8))
        
        t2 = time.time()
        status_text.text("⏱️ Step 3/5: Finding geometry contours...")
        scale = self.target_width / work_img.shape[1]
        contours, hierarchy = cv2.findContours(clean, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        debug_img = cv2.cvtColor(clean, cv2.COLOR_GRAY2RGB)
        
        if not contours: return None, None, None

        t3 = time.time()
        total_contours = len(contours)
        
        model = cq.Workplane("XY")
        processed_count = 0
        
        for i, cnt in enumerate(contours):
            if i % 5 == 0:
                progress_bar.progress(i / total_contours)
                status_text.text(f"⚙️ Fast-Extruding shape {i}/{total_contours}...")

            # 1. AREA FILTER: Drop tiny dust and micro-details
            if cv2.contourArea(cnt) < self.min_area: 
                continue
            
            # 2. VERTEX REDUCTION: Smooth the lines to drastically speed up 3D math
            epsilon = self.smoothness * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            # Skip if the shape got smoothed out of existence
            if len(approx) < 3: 
                continue
            
            pts = [(p[0][0] * scale, -p[0][1] * scale) for p in approx]
            is_hole = hierarchy[0][i][3] != -1
            
            # Draw preview
            cv2.drawContours(debug_img, [approx], -1, (0, 255, 0) if not is_hole else (255, 0, 0), 1)

            try:
                if not is_hole:
                    model = model.polyline(pts).close().extrude(self.height)
                else:
                    hole = cq.Workplane("XY").polyline(pts).close().extrude(self.height)
                    model = model.cut(hole)
                processed_count += 1
            except Exception as e:
                continue
                
        progress_bar.progress(1.0)
        t4 = time.time()
        status_text.text(f"⏱️ Step 5/5: Exporting files... (3D Math took {t4-t3:.2f}s)")
        
        step_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".step")
        stl_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".stl")
        
        cq.exporters.export(model, step_tmp.name)
        cq.exporters.export(model, stl_tmp.name)
        
        t5 = time.time()
        status_text.text(f"✅ Complete! Rendered {processed_count} parts in {t5-t0:.2f} seconds.")
        
        return step_tmp.name, stl_tmp.name, debug_img

# --- UI Setup ---
st.set_page_config(page_title="Window CAD Generator", layout="wide")
st.title("🪟 Automated Window CAD Pipeline")
st.write("Upload a 2D sketch or photo to generate a scalable 3D printable STEP file.")

with st.sidebar:
    st.header("1. Dimensions")
    h = st.slider("Thickness (mm)", 1, 50, 5)
    w = st.number_input("Target Width (mm)", value=200.0)
    
    st.header("2. Optimization (Speed)")
    st.write("Adjust these to dramatically speed up generation time.")
    min_area = st.slider("Ignore small details (Area)", 10, 1000, 100)
    smoothness = st.slider("Line Smoothing Factor", 0.0, 0.005, 0.001, format="%.4f")
    
    st.header("3. Advanced")
    use_deskew = st.toggle("Fix Perspective (Deskew)", value=False)
    uploaded_file = st.file_uploader("Upload Drawing", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    proc = WindowProcessor(h, w, use_deskew, min_area, smoothness)
    step_file, stl_file, debug_preview = proc.process(uploaded_file.getvalue(), progress_bar, status_text)

    if step_file:
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("1. Detected Paths")
            st.image(debug_preview, caption="Blue = Holes | Green = Solid Frame")
            with open(step_file, "rb") as f:
                st.download_button("💾 Download STEP File", f, "window_model.step", type="primary")
                
        with c2:
            st.subheader("2. 3D Preview")
            try:
                mesh = trimesh.load_mesh(stl_file)
                fig = go.Figure(data=[go.Mesh3d(
                    x=mesh.vertices[:, 0], y=mesh.vertices[:, 1], z=mesh.vertices[:, 2],
                    i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2],
                    color='lightblue', flatshading=True,
                    lighting=dict(ambient=0.5, diffuse=0.8, fresnel=0.5, specular=0.5, roughness=0.1)
                )])
                fig.update_layout(
                    scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='data'),
                    margin=dict(l=0, r=0, b=0, t=0), height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not render 3D preview: {e}")
            finally:
                if os.path.exists(step_file): os.unlink(step_file)
                if os.path.exists(stl_file): os.unlink(stl_file)
