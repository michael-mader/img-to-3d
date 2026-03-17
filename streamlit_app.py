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
        
        img_area = img.shape[0] * img.shape[1]

        for c in contours:
            area = cv2.contourArea(c)
            # SAFEGUARD: If the detected contour is >90% of the image size, 
            # it's just the image border. Ignore it.
            if area > 0.90 * img_area:
                continue
                
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

    def process(self, image_bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Add a 10px white border to prevent edge-bleeding
        img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        
        # 1. Deskew if enabled
        work_img = self._apply_deskew(img) if self.do_deskew else img
        
        # 2. Image Prep & Morphology
        gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        # Use a slightly smaller kernel (2x2) for highly detailed images like the Easter one
        clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8))
        
        # 3. Scaling & Contours
        scale = self.target_width / work_img.shape[1]
        contours, hierarchy = cv2.findContours(clean, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        debug_img = cv2.cvtColor(clean, cv2.COLOR_GRAY2RGB)
        model = cq.Workplane("XY")
        
        if not contours: return None, None, None

        for i, cnt in enumerate(contours):
            # Lowered the minimum area from 150 to 50 so it doesn't delete small details (like the bunny's eye)
            if cv2.contourArea(cnt) < 50: continue
            
            # Map pixels to millimeters, invert Y for CAD coordinates
            pts = [(p[0][0] * scale, -p[0][1] * scale) for p in cnt]
            is_hole = hierarchy[0][i][3] != -1
            
            # Draw preview: Green = Solid, Blue = Hole
            cv2.drawContours(debug_img, [cnt], -1, (0, 255, 0) if not is_hole else (255, 0, 0), 1)

            # 4. Extrude Geometry
            try:
                if not is_hole:
                    model = model.polyline(pts).close().extrude(self.height)
                else:
                    hole = cq.Workplane("XY").polyline(pts).close().extrude(self.height)
                    model = model.cut(hole)
            except Exception as e:
                # If a specific tiny shape fails to extrude, skip it rather than crashing the whole app
                print(f"Skipped complex contour: {e}")
                continue
        
        # 5. Export Temp Files
        step_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".step")
        stl_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".stl")
        
        cq.exporters.export(model, step_tmp.name)
        cq.exporters.export(model, stl_tmp.name)
        
        return step_tmp.name, stl_tmp.name, debug_img
