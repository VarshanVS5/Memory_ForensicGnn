import { useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Text, Line } from "@react-three/drei";
import * as THREE from "three";
import type { GraphSchema, AttentionEdge } from "./api";

function fibonacciSphere(n: number, radius: number): [number, number, number][] {
  const points: [number, number, number][] = [];
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < n; i++) {
    const y = 1 - (i / (n - 1)) * 2;
    const r = Math.sqrt(1 - y * y);
    const theta = goldenAngle * i;
    points.push([Math.cos(theta) * r * radius, y * radius, Math.sin(theta) * r * radius]);
  }
  return points;
}

function Node({
  position,
  id,
  label,
  active,
  selected,
  onClick,
}: {
  position: [number, number, number];
  id: string;
  label: string;
  active: boolean;
  selected: boolean;
  onClick: (id: string) => void;
}) {
  const [hovered, setHovered] = useState(false);
  const isHighlighted = active || hovered || selected;

  return (
    <group
      position={position}
      onClick={(e) => {
        e.stopPropagation();
        onClick(id);
      }}
      onPointerOver={(e) => {
        e.stopPropagation();
        setHovered(true);
      }}
      onPointerOut={() => setHovered(false)}
    >
      {/* Halo glow ring for active/selected nodes */}
      {isHighlighted && (
        <mesh>
          <sphereGeometry args={[active ? 0.42 : 0.38, 24, 24]} />
          <meshBasicMaterial
            color={active ? "#f43f5e" : "#6366f1"}
            transparent
            opacity={0.25}
          />
        </mesh>
      )}

      {/* Core Node Sphere */}
      <mesh>
        <sphereGeometry args={[active ? 0.32 : 0.24, 28, 28]} />
        <meshStandardMaterial
          color={active ? "#e11d48" : selected ? "#4f46e5" : "#6366f1"}
          emissive={active ? "#f43f5e" : selected ? "#6366f1" : "#818cf8"}
          emissiveIntensity={active ? 0.8 : selected ? 0.6 : 0.3}
          roughness={0.25}
          metalness={0.3}
        />
      </mesh>

      {/* Crisp Light-Theme Node Text Label */}
      <Text
        position={[0, 0.44, 0]}
        fontSize={0.17}
        fontWeight={700}
        color={active ? "#be123c" : "#0f172a"}
        anchorX="center"
        anchorY="bottom"
        outlineWidth={0.02}
        outlineColor="#ffffff"
      >
        {label}
      </Text>
    </group>
  );
}

function Edge({
  a,
  b,
  weight,
  maxWeight,
}: {
  a: [number, number, number];
  b: [number, number, number];
  weight: number | null;
  maxWeight: number;
}) {
  const norm = weight != null && maxWeight > 0 ? weight / maxWeight : 0;
  const color = weight == null ? "#cbd5e1" : norm > 0.6 ? "#e11d48" : norm > 0.3 ? "#f59e0b" : "#64748b";
  const width = weight == null ? 1.5 : 2 + norm * 5;
  const opacity = weight == null ? 0.45 : 0.4 + norm * 0.55;

  return <Line points={[a, b]} color={color} lineWidth={width} transparent opacity={opacity} />;
}

function Scene({
  schema,
  attentionEdges,
  selectedNode,
  onSelectNode,
}: {
  schema: GraphSchema;
  attentionEdges: AttentionEdge[];
  selectedNode: string | null;
  onSelectNode: (id: string) => void;
}) {
  const groupRef = useRef<THREE.Group>(null);
  const positions = useMemo(() => {
    const pts = fibonacciSphere(schema.nodes.length, 2.5);
    const map: Record<string, [number, number, number]> = {};
    schema.nodes.forEach((n, i) => (map[n.id] = pts[i]));
    return map;
  }, [schema]);

  const edgeWeightMap = useMemo(() => {
    const map: Record<string, number> = {};
    attentionEdges.forEach((e) => {
      map[[e.source, e.target].sort().join("|")] = e.weight;
    });
    return map;
  }, [attentionEdges]);

  const maxWeight = useMemo(
    () => Math.max(0, ...attentionEdges.map((e) => e.weight)),
    [attentionEdges]
  );

  useFrame((_, delta) => {
    if (groupRef.current) groupRef.current.rotation.y += delta * 0.06;
  });

  const activeNodes = useMemo(() => {
    const s = new Set<string>();
    const sorted = [...attentionEdges].sort((a, b) => b.weight - a.weight).slice(0, 3);
    sorted.forEach((e) => {
      s.add(e.source);
      s.add(e.target);
    });
    return s;
  }, [attentionEdges]);

  return (
    <group ref={groupRef}>
      {schema.edges.map((e, i) => {
        const key = [e.source, e.target].sort().join("|");
        return (
          <Edge
            key={i}
            a={positions[e.source]}
            b={positions[e.target]}
            weight={edgeWeightMap[key] ?? null}
            maxWeight={maxWeight}
          />
        );
      })}
      {schema.nodes.map((n) => (
        <Node
          key={n.id}
          id={n.id}
          position={positions[n.id]}
          label={n.label}
          active={activeNodes.has(n.id)}
          selected={selectedNode === n.id}
          onClick={onSelectNode}
        />
      ))}
    </group>
  );
}

export default function Graph3D({
  schema,
  attentionEdges,
  selectedNode = null,
  onSelectNode = () => {},
}: {
  schema: GraphSchema | null;
  attentionEdges: AttentionEdge[];
  selectedNode?: string | null;
  onSelectNode?: (id: string) => void;
}) {
  if (!schema) return null;

  return (
    <Canvas camera={{ position: [0, 0, 6.8], fov: 48 }} style={{ width: "100%", height: "100%" }} dpr={[1, 2]}>
      {/* Soft light Slate background color */}
      <color attach="background" args={["#f8fafc"]} />
      <ambientLight intensity={1.2} />
      <directionalLight position={[10, 15, 10]} intensity={1.8} color="#ffffff" />
      <directionalLight position={[-10, -10, -5]} intensity={0.8} color="#e0e7ff" />
      <pointLight position={[0, 8, 5]} intensity={30} color="#6366f1" />
      
      <Scene
        schema={schema}
        attentionEdges={attentionEdges}
        selectedNode={selectedNode}
        onSelectNode={onSelectNode}
      />
      <OrbitControls enablePan={false} autoRotate={false} minDistance={3.5} maxDistance={11} />
    </Canvas>
  );
}

