import { Suspense, useEffect, useMemo, useRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import {
  ContactShadows,
  Html,
  OrbitControls,
  useGLTF,
  useTexture,
} from "@react-three/drei";
import * as THREE from "three";

const MODEL_PATH = "/model.glb";

/* -------------------------------------------------------
   PRINTER POSITION
------------------------------------------------------- */

const MODEL_SCALE = 3.55;

const MODEL_POSITION = [
  0,
  -0.55,
  0,
];

/* -------------------------------------------------------
   RECEIPT POSITION

   PRESERVED EXACTLY as requested.
------------------------------------------------------- */

const PAPER_POSITION = [
  0,
  -0.23,
  0.47,
];

/* -------------------------------------------------------
   RECEIPT SIZE
------------------------------------------------------- */

/*
 * Width should approximately match the printer mouth.
 *
 * Increase:
 *   0.66 / 0.68
 *
 * if the receipt is too narrow.
 *
 * Decrease:
 *   0.58 / 0.56
 *
 * if it is too wide.
 */
const PAPER_WIDTH = 0.62;

/*
 * Number of subdivisions.
 * More segments = smoother paper bending.
 */
const LENGTH_SEGMENTS = 40;
const WIDTH_SEGMENTS = 8;

/*
 * How much of the paper is inside the printer.
 */
const INSIDE_RATIO = 0.5;

/*
 * Maximum paper bend.
 */
const SAG_AMOUNT = 0.22;

/* -------------------------------------------------------
   PRINTER MODEL
------------------------------------------------------- */

function PrinterModel({ printing }) {
  const group = useRef();
  const { pointer } = useThree();

  const { scene } = useGLTF(MODEL_PATH);

  /*
   * Hide the original paper/sheet that came with
   * the downloaded printer model.
   *
   * The uploaded receipt is rendered separately by
   * ReceiptPaper(), so we don't need the GLB's paper.
   */
  useEffect(() => {
  scene.traverse((child) => {
    if (!child.isMesh) return;

    const materialName =
      child.material?.name?.toLowerCase() || "";

    const meshName =
      child.name?.toLowerCase() || "";

    /*
     * The GLB's cream mesh is the built-in
     * white paper/board we want to remove.
     */
    if (
      materialName === "cream" ||
      meshName.endsWith("_2")
    ) {
      child.visible = false;
    }
  });
}, [scene]);

  useFrame((state) => {
    if (!group.current) {
      return;
    }

    const time =
      state.clock.getElapsedTime();

    /*
     * Subtle mouse-controlled product movement.
     */
    const targetRotationY =
      pointer.x * 0.028;

    const targetRotationX =
      pointer.y * 0.012;

    group.current.rotation.y =
      THREE.MathUtils.lerp(
        group.current.rotation.y,
        targetRotationY,
        0.035
      );

    group.current.rotation.x =
      THREE.MathUtils.lerp(
        group.current.rotation.x,
        targetRotationX,
        0.035
      );

    /*
     * Tiny physical vibration while processing.
     */
    const vibrationX = printing
      ? Math.sin(time * 78) * 0.004
      : 0;

    const vibrationZ = printing
      ? Math.sin(time * 65) * 0.002
      : 0;

    const idleY =
      Math.sin(time * 0.75) * 0.008;

    group.current.position.x =
      MODEL_POSITION[0] +
      vibrationX;

    group.current.position.y =
      MODEL_POSITION[1] +
      idleY;

    group.current.position.z =
      MODEL_POSITION[2] +
      vibrationZ;
  });

  return (
    <group
      ref={group}
      position={MODEL_POSITION}
      scale={MODEL_SCALE}
    >
      <primitive object={scene} />
    </group>
  );
}

/* -------------------------------------------------------
   CURVED PAPER GEOMETRY
------------------------------------------------------- */

/*
 * Coordinate system:
 *
 * X = paper width
 * Z = paper feed direction
 * Y = vertical bend
 *
 * The printer mouth is Z = 0.
 *
 * Negative Z:
 *   paper is INSIDE the printer.
 *
 * Positive Z:
 *   paper is OUTSIDE the printer.
 *
 * This makes the layout much easier to control.
 */
function createReceiptGeometry(width, length) {
  const geometry = new THREE.BufferGeometry();

  const positions = [];
  const normals = [];
  const uvs = [];
  const indices = [];

  const halfWidth = width / 2;

  const insideLength =
    length * INSIDE_RATIO;

  const outsideLength =
    length * (1 - INSIDE_RATIO);

  for (
    let row = 0;
    row <= LENGTH_SEGMENTS;
    row += 1
  ) {
    const v =
      row / LENGTH_SEGMENTS;

    /*
     * z moves from:
     *
     * -insideLength
     *
     * to
     *
     * +outsideLength
     */
    const z = THREE.MathUtils.lerp(
      -insideLength,
      outsideLength,
      v
    );

    /*
     * Paper is flat while it is inside
     * the printer.
     */
    let bend = 0;

    /*
     * The bend starts exactly at the mouth.
     */
    if (z > 0) {
      const t =
        z / outsideLength;

      /*
       * Smooth ease-in.
       *
       * At the mouth:
       *   0
       *
       * At the free end:
       *   1
       */
      const smoothT =
        t * t * (3 - 2 * t);

      /*
       * Gravity pulls the free end down.
       */
      bend =
        -SAG_AMOUNT *
        smoothT *
        smoothT;
    }

    for (
      let column = 0;
      column <= WIDTH_SEGMENTS;
      column += 1
    ) {
      const u =
        column / WIDTH_SEGMENTS;

      const x = THREE.MathUtils.lerp(
        -halfWidth,
        halfWidth,
        u
      );

      /*
       * The centre of the paper hangs
       * naturally.
       *
       * The sides sag just a tiny bit more,
       * giving it a soft real-paper appearance.
       */
      const edgeDistance =
        Math.abs(u - 0.5) * 2;

      const sideSag =
        z > 0
          ? -0.018 *
            Math.pow(
              edgeDistance,
              1.7
            ) *
            smoothFactor(z, outsideLength)
          : 0;

      const y = bend + sideSag;

      positions.push(
        x,
        y,
        z
      );

      normals.push(
        0,
        1,
        0
      );

      /*
       * Keep the image oriented normally.
       */
      uvs.push(
        u,
        1 - v
      );
    }
  }

  const columns =
    WIDTH_SEGMENTS + 1;

  for (
    let row = 0;
    row < LENGTH_SEGMENTS;
    row += 1
  ) {
    for (
      let column = 0;
      column < WIDTH_SEGMENTS;
      column += 1
    ) {
      const a =
        row * columns +
        column;

      const b =
        a + 1;

      const c =
        a + columns;

      const d =
        c + 1;

      indices.push(
        a,
        c,
        b
      );

      indices.push(
        b,
        c,
        d
      );
    }
  }

  geometry.setIndex(indices);

  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(
      positions,
      3
    )
  );

  geometry.setAttribute(
    "normal",
    new THREE.Float32BufferAttribute(
      normals,
      3
    )
  );

  geometry.setAttribute(
    "uv",
    new THREE.Float32BufferAttribute(
      uvs,
      2
    )
  );

  geometry.computeVertexNormals();
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();

  return geometry;
}

/*
 * Small helper used for the paper edge sag.
 */
function smoothFactor(z, outsideLength) {
  if (z <= 0) {
    return 0;
  }

  const t = THREE.MathUtils.clamp(
    z / outsideLength,
    0,
    1
  );

  return t * t * (3 - 2 * t);
}

/* -------------------------------------------------------
   RECEIPT PAPER
------------------------------------------------------- */

function ReceiptPaper({
  preview,
  printing,
}) {
  const paperRef = useRef();

  const texture =
    useTexture(preview);

  /*
   * Determine the original image aspect ratio.
   */
  const imageAspect =
    useMemo(() => {
      const image = texture?.image;

      if (
        !image ||
        !image.width ||
        !image.height
      ) {
        return 0.65;
      }

      return (
        image.width /
        image.height
      );
    }, [texture]);

  /*
   * Preserve the uploaded receipt's proportions.
   */
  const paperLength =
    useMemo(() => {
      const naturalLength =
        PAPER_WIDTH /
        imageAspect;

      return THREE.MathUtils.clamp(
        naturalLength,
        0.95,
        1.65
      );
    }, [imageAspect]);

  /*
   * Build curved receipt geometry.
   */
  const geometry =
    useMemo(() => {
      return createReceiptGeometry(
        PAPER_WIDTH,
        paperLength
      );
    }, [paperLength]);

  /*
   * Dispose old geometry when replaced.
   */
  useEffect(() => {
    return () => {
      geometry.dispose();
    };
  }, [geometry]);

  /*
   * Texture setup.
   */
  useEffect(() => {
    if (!texture) {
      return;
    }

    texture.colorSpace =
      THREE.SRGBColorSpace;

    texture.anisotropy = 4;

    texture.needsUpdate = true;
  }, [texture]);

  useFrame((state) => {
    if (!paperRef.current) {
      return;
    }

    const time =
      state.clock.getElapsedTime();

    /*
     * Tiny roller vibration.
     */
    const vibrationX = printing
      ? Math.sin(time * 11) *
        0.0025
      : 0;

    paperRef.current.position.x =
      THREE.MathUtils.lerp(
        paperRef.current.position.x,
        vibrationX,
        0.08
      );

    /*
     * Tiny forward/back movement while
     * the printer is working.
     */
    const feedMovement = printing
      ? Math.sin(time * 5.5) *
        0.012
      : 0;

    paperRef.current.position.z =
      THREE.MathUtils.lerp(
        paperRef.current.position.z,
        PAPER_POSITION[2] +
          feedMovement,
        0.06
      );
  });

  return (
    <mesh
      ref={paperRef}
      geometry={geometry}
      position={[
        PAPER_POSITION[0],
        PAPER_POSITION[1],
        PAPER_POSITION[2],
      ]}
      castShadow
      receiveShadow
    >
      <meshStandardMaterial
        map={texture}
        roughness={0.96}
        metalness={0}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

/* -------------------------------------------------------
   SCENE
------------------------------------------------------- */

function Scene({
  preview,
  printing,
}) {
  return (
    <>
      <color
        attach="background"
        args={["#e9edf2"]}
      />

      <ambientLight
        intensity={2.15}
      />

      <directionalLight
        position={[4, 6, 5]}
        intensity={3.5}
        castShadow
        shadow-mapSize-width={1024}
        shadow-mapSize-height={1024}
      />

      <directionalLight
        position={[-4, 3, 2]}
        intensity={1.25}
      />

      <spotLight
        position={[1, 4, 4]}
        angle={0.45}
        penumbra={0.8}
        intensity={1.7}
      />

      <PrinterModel
        printing={printing}
      />

      {preview && (
        <ReceiptPaper
          preview={preview}
          printing={printing}
        />
      )}

      <ContactShadows
        position={[0, -1.13, 0]}
        opacity={0.23}
        scale={4}
        blur={2.2}
        far={3}
      />

      <OrbitControls
        makeDefault
        enableZoom={false}
        enablePan={false}
        enableDamping
        dampingFactor={0.06}
        target={[0, -0.38, 0]}
        minPolarAngle={Math.PI / 2.45}
        maxPolarAngle={Math.PI / 2.02}
        minAzimuthAngle={-0.3}
        maxAzimuthAngle={0.3}
      />
    </>
  );
}

/* -------------------------------------------------------
   LOADER
------------------------------------------------------- */

function LoadingPrinter() {
  return (
    <Html center>
      <div className="printer-loader">
        Loading printer...
      </div>
    </Html>
  );
}

/* -------------------------------------------------------
   MAIN COMPONENT
------------------------------------------------------- */

export default function Printer3D({
  preview,
  printing,
}) {
  return (
    <div className="printer-canvas">
      <Canvas
        shadows
        dpr={[1, 1.35]}
        camera={{
          /*
           * Lower camera angle so the printer
           * sits around the centre of the scene.
           */
          position: [
            2.65,
            0.85,
            3.45,
          ],
          fov: 31,
        }}
        gl={{
          antialias: true,
          alpha: false,
          powerPreference:
            "high-performance",
        }}
      >
        <Suspense
          fallback={
            <LoadingPrinter />
          }
        >
          <Scene
            preview={preview}
            printing={printing}
          />
        </Suspense>
      </Canvas>
    </div>
  );
}

useGLTF.preload(
  MODEL_PATH
);