import streamlit as st
import cv2
import numpy as np
import cadquery as cq
import tempfile
import os
import trimesh
import plotly.graph_objects as go

class WindowProcessor:
    def __init__(self, height, target_width, do_deskew):
        self.height = height
        self.target_width = target_width
        self.do_deskew = do_deskew

    def _apply_deskew(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 50, 200)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for c in contours:
            approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype("float32")
                # Sort points: top-left, top-right, bottom-right, bottom-left
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

    def process(self, image_bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 1. Deskew if enabled
        work_img = self._apply_deskew(img) if self.do_deskew else img
        
        # 2. Image Prep & Morphology
        gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
        
        # 3. Scaling & Contours
        scale = self.target_width / work_img.shape[1]
        contours, hierarchy = cv2.findContours(clean, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        debug_img = cv2.cvtColor(clean, cv2.COLOR_GRAY2RGB)
        model = cq.Workplane("XY")
        
        if not contours: return None, None, None

        for i, cnt in enumerate(contours):
            if cv2.contourArea(cnt) < 150: continue
            
            # Map pixels to millimeters, invert Y for CAD coordinates
            pts = [(p[0][0] * scale, -p[0][1] * scale) for p in cnt]
            is_hole = hierarchy[0][i][3] != -1
            
            # Draw preview: Green = Solid, Blue = Hole
            cv2.drawContours(debug_img, [cnt], -1, (0, 255, 0) if not is_hole else (255, 0, 0), 2)

            # 4. Extrude Geometry
            if not is_hole:
                model = model.polyline(pts).close().extrude(self.height)
            else:
                hole = cq.Workplane("XY").polyline(pts).close().extrude(self.height)
                model = model.cut(hole)
        
        # 5. Export Temp Files
        step_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".step")
        stl_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".stl")
        model.exportStep(step_tmp.name)
        model.exportStl(stl_tmp.name)
        
        return step_tmp.name, stl_tmp.name, debug_img

# --- UI Setup ---
st.set_page_config(page_title="Window CAD Generator", layout="wide")
st.title("🪟 Automated Window CAD Pipeline")
st.write("Upload a 2D sketch or photo to generate a scalable 3D printable STEP file.")

with st.sidebar:
    st.header("Controls")
    h = st.slider("Thickness (mm)", 1, 50, 5)
    w = st.number_input("Target Width (mm)", value=200.0)
    use_deskew = st.toggle("Fix Perspective (Deskew)", value=True)
    uploaded_file = st.file_uploader("Upload Drawing", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    with st.spinner("Analyzing geometry and building 3D model..."):
        proc = WindowProcessor(h, w, use_deskew)
        step_file, stl_file, debug_preview = proc.process(uploaded_file.getvalue())

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
                # Load the STL with trimesh
                mesh = trimesh.load_mesh(stl_file)
                
                # Create a Plotly 3D Mesh
                fig = go.Figure(data=[go.Mesh3d(
                    x=mesh.vertices[:, 0],
                    y=mesh.vertices[:, 1],
                    z=mesh.vertices[:, 2],
                    i=mesh.faces[:, 0],
                    j=mesh.faces[:, 1],
                    k=mesh.faces[:, 2],
                    color='lightblue',
                    flatshading=True,
                    lighting=dict(ambient=0.5, diffuse=0.8, fresnel=0.5, specular=0.5, roughness=0.1)
                )])
                
                # Format the layout to look like a CAD viewer
                fig.update_layout(
                    scene=dict(
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False),
                        zaxis=dict(visible=False),
                        aspectmode='data'
                    ),
                    margin=dict(l=0, r=0, b=0, t=0),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not render 3D preview: {e}")
            
            finally:
                # Cleanup temporary files
                if os.path.exists(step_file):
                    os.unlink(step_file)
                if os.path.exists(stl_file):
                    os.unlink(stl_file)