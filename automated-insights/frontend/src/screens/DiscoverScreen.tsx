import { Button, Container, Typography, Box, Paper, CircularProgress, useTheme } from '@mui/material'
import { useNavigate } from 'react-router-dom'
import React, { useState, useEffect, useRef, useMemo } from 'react';

import { createConversation, createMessageAsync } from '../graphql/mutations';
import { getConversation } from '../graphql/queries';
import { recieveMessageChunkAsync } from "../graphql/subscriptions";
import { generateClient } from 'aws-amplify/data';

const client = generateClient({
    authMode: "userPool",
  });

// Interface for entity relationship data
interface EntityRelationship {
  entities: Entity[];
  relationships: Relationship[];
}

interface Entity {
  name: string;
  type?: string;
  attributes: EntityAttribute[];
}

interface EntityAttribute {
  name: string;
  type: string;
  key?: string;
}

interface Relationship {
  from: string;
  to: string;
  type: string;
  fromAttribute?: string;
  toAttribute?: string;
}

interface Message {
  __typename: "Message";
  sender: string;
  message: string;
  createdAt: string;
}

// Function to parse the non-standard format to valid JSON
const parseEntityRelationshipData = (data: string): EntityRelationship => {
  try {
    // Check if it's already valid JSON
    try {
      return JSON.parse(data);
    } catch (e) {
      // Not valid JSON, continue with custom parsing
      console.log("Not valid JSON, attempting custom parsing");
    }

    // Clean up the string - remove outer quotes if present
    let cleanData = data;
    if (cleanData.startsWith('"') && cleanData.endsWith('"')) {
      cleanData = cleanData.substring(1, cleanData.length - 1);
    }
    
    // Log the data for debugging
    console.log("Data to parse:", cleanData);
    
    // Direct regex extraction approach - more reliable for this specific format
    try {
      console.log("Using direct regex extraction approach");
      const result: EntityRelationship = { entities: [], relationships: [] };
      
      // Extract entities section
      const entitiesMatch = cleanData.match(/entities=\[(.*?)\],\s*relationships=/s);
      if (entitiesMatch && entitiesMatch[1]) {
        const entitiesStr = entitiesMatch[1];
        
        // Extract each entity using regex - with explicit escaping to prevent changes
        const entityRegex = new RegExp("\\{name=([^,]+),\\s*attributes=\\[(.*?)\\]\\}", "gs");
        const entityMatches = [...entitiesStr.matchAll(entityRegex)];
        
        for (const match of entityMatches) {
          const entityName = match[1].trim();
          const attributesStr = match[2];
          const attributes: EntityAttribute[] = [];
          
          // Extract attributes using regex
          const attrRegex = /\{name=([^,]+),\s*type=([^,\}]+)(?:,\s*key=([^,\}]+))?\}/g;
          const attrMatches = [...attributesStr.matchAll(attrRegex)];
          
          for (const attrMatch of attrMatches) {
            attributes.push({
              name: attrMatch[1].trim(),
              type: attrMatch[2].trim(),
              key: attrMatch[3] ? attrMatch[3].trim() : undefined
            });
          }
          
          result.entities.push({
            name: entityName,
            type: 'table',
            attributes
          });
        }
      }
      
      // Extract relationships section
      const relsMatch = cleanData.match(/relationships=\[(.*?)\](?:\s*\})?$/s);
      if (relsMatch && relsMatch[1]) {
        const relsStr = relsMatch[1];
        
        // Extract each relationship using regex
        const relRegex = /\{from=([^,]+),\s*to=([^,]+),\s*type=([^,]+),\s*fromAttribute=([^,]+),\s*toAttribute=([^,\}]+)\}/g;
        const relMatches = [...relsStr.matchAll(relRegex)];
        
        for (const match of relMatches) {
          result.relationships.push({
            from: match[1].trim(),
            to: match[2].trim(),
            type: match[3].trim(),
            fromAttribute: match[4].trim(),
            toAttribute: match[5].trim()
          });
        }
      }
      
      // If we have successfully extracted data, return it
      if (result.entities.length > 0 || result.relationships.length > 0) {
        console.log("Direct regex extraction successful");
        return result;
      }
      
      throw new Error("Failed to extract data with regex");
    } catch (regexError) {
      console.error("Error in direct regex extraction:", regexError);
      
      // Try another approach - handle the specific format with manual string manipulation
      try {
        console.log("Attempting manual string manipulation approach");
        
        // Replace = with : for key-value pairs
        let jsonStr = cleanData.replace(/(\w+)=/g, '"$1":');
        
        // Handle decimal values with commas - this is the problematic part
        jsonStr = jsonStr.replace(/DECIMAL\((\d+),(\d+)\)/g, 'DECIMAL($1.$2)');
        
        // Add quotes around unquoted string values
        jsonStr = jsonStr.replace(/:([^",\{\}\[\]\s][^,\{\}\[\]]*)/g, ':"$1"');
        
        // Ensure all property names are quoted
        jsonStr = jsonStr.replace(/\{([^{}]*?)\}/g, (match) => {
          return match.replace(/([a-zA-Z0-9_]+):/g, '"$1":');
        });
        
        // Wrap the whole thing in curly braces if not already
        if (!jsonStr.startsWith('{')) {
          jsonStr = '{' + jsonStr + '}';
        }
        
        console.log("Manually transformed data:", jsonStr);
        
        // Try to parse the transformed string
        const parsed = JSON.parse(jsonStr);
        
        // Transform the data to match our interface
        const result: EntityRelationship = {
          entities: [],
          relationships: []
        };
        
        if (parsed.entities) {
          result.entities = parsed.entities.map((entity: any) => ({
            name: entity.name,
            type: entity.type || 'table',
            attributes: Array.isArray(entity.attributes) 
              ? entity.attributes.map((attr: any) => ({
                  name: attr.name,
                  type: attr.type,
                  key: attr.key
                }))
              : []
          }));
        }
        
        if (parsed.relationships) {
          result.relationships = parsed.relationships.map((rel: any) => ({
            from: rel.from,
            to: rel.to,
            type: rel.type,
            fromAttribute: rel.fromAttribute,
            toAttribute: rel.toAttribute
          }));
        }
        
        return result;
      } catch (manualError) {
        console.error("Error in manual string manipulation:", manualError);
        
        // Last resort - use a very specific parser for the exact format we're seeing
        console.log("Using last resort specific parser");
        
        // Split the string into entities and relationships sections
        const parts = cleanData.split('relationships=');
        if (parts.length !== 2) {
          throw new Error("Could not split data into entities and relationships sections");
        }
        
        const entitiesStr = parts[0].replace('entities=', '').trim();
        const relationshipsStr = parts[1].trim();
        
        const result: EntityRelationship = { entities: [], relationships: [] };
        
        // Parse entities
        // Remove the outer brackets and split by closing curly brace followed by comma and opening curly brace
        const entityStrings = entitiesStr
          .replace(/^\[/, '')
          .replace(/\],?$/, '')
          .split(/\},\s*\{/);
          
        for (let i = 0; i < entityStrings.length; i++) {
          let entityStr = entityStrings[i];
          if (!entityStr.startsWith('{')) entityStr = '{' + entityStr;
          if (!entityStr.endsWith('}')) entityStr = entityStr + '}';
          
          // Extract entity name
          const nameMatch = entityStr.match(/name=([^,]+)/);
          if (!nameMatch) continue;
          
          const name = nameMatch[1].trim();
          
          // Extract attributes
          const attributesMatch = entityStr.match(/attributes=\[(.*?)\]/s);
          if (!attributesMatch) continue;
          
          const attributesStr = attributesMatch[1];
          const attributeStrings = attributesStr.split(/\},\s*\{/);
          
          const attributes: EntityAttribute[] = [];
          
          for (let j = 0; j < attributeStrings.length; j++) {
            let attrStr = attributeStrings[j];
            if (!attrStr.startsWith('{')) attrStr = '{' + attrStr;
            if (!attrStr.endsWith('}')) attrStr = attrStr + '}';
            
            const attrNameMatch = attrStr.match(/name=([^,]+)/);
            const attrTypeMatch = attrStr.match(/type=([^,\}]+)/);
            const attrKeyMatch = attrStr.match(/key=([^,\}]+)/);
            
            if (attrNameMatch && attrTypeMatch) {
              attributes.push({
                name: attrNameMatch[1].trim(),
                type: attrTypeMatch[1].trim(),
                key: attrKeyMatch ? attrKeyMatch[1].trim() : undefined
              });
            }
          }
          
          result.entities.push({
            name,
            type: 'table',
            attributes
          });
        }
        
        // Parse relationships
        // Remove the outer brackets and split by closing curly brace followed by comma and opening curly brace
        const relationshipStrings = relationshipsStr
          .replace(/^\[/, '')
          .replace(/\]\}?$/, '')
          .split(/\},\s*\{/);
          
        for (let i = 0; i < relationshipStrings.length; i++) {
          let relStr = relationshipStrings[i];
          if (!relStr.startsWith('{')) relStr = '{' + relStr;
          if (!relStr.endsWith('}')) relStr = relStr + '}';
          
          const fromMatch = relStr.match(/from=([^,]+)/);
          const toMatch = relStr.match(/to=([^,]+)/);
          const typeMatch = relStr.match(/type=([^,]+)/);
          const fromAttrMatch = relStr.match(/fromAttribute=([^,]+)/);
          const toAttrMatch = relStr.match(/toAttribute=([^,\}]+)/);
          
          if (fromMatch && toMatch && typeMatch && fromAttrMatch && toAttrMatch) {
            result.relationships.push({
              from: fromMatch[1].trim(),
              to: toMatch[1].trim(),
              type: typeMatch[1].trim(),
              fromAttribute: fromAttrMatch[1].trim(),
              toAttribute: toAttrMatch[1].trim()
            });
          }
        }
        
        return result;
      }
    }
  } catch (error) {
    console.error('Error parsing entity relationship data:', error);
    throw new Error('Failed to parse entity relationship data');
  }
};

// New ERDiagram component for visual representation
const ERDiagram = ({ entityData }: { entityData: EntityRelationship }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const diagramRef = useRef<HTMLDivElement>(null);
  const theme = useTheme();
  const [scale, setScale] = useState(1.0);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  
  // Calculate positions for entities in a grid layout
  const entityPositions = useMemo(() => {
    const entities = entityData.entities;
    const positions: Record<string, { x: number, y: number, width: number, height: number }> = {};
    
    // Calculate grid layout
    const numEntities = entities.length;
    const aspectRatio = 16/9;
    const cols = Math.max(2, Math.ceil(Math.sqrt(numEntities * (aspectRatio/3))));
    
    // Calculate grid cell size - increase cell height to prevent overlap
    const cellWidth = 700;
    const cellHeight = 650; // Increased from 600 to 650 for more vertical space
    
    // Position entities in a grid layout with margin
    entities.forEach((entity, index) => {
      const row = Math.floor(index / cols);
      const col = index % cols;
      
      // Find the longest attribute name and type for width calculation
      let maxAttributeLength = 0;
      let maxTypeLength = 0;
      
      entity.attributes.forEach(attr => {
        // Calculate total length including key indicators
        let nameLength = attr.name.length;
        if (attr.key === 'PK') nameLength += 4;
        if (attr.key === 'FK') nameLength += 4;
        
        if (nameLength > maxAttributeLength) {
          maxAttributeLength = nameLength;
        }
        
        if (attr.type.length > maxTypeLength) {
          maxTypeLength = attr.type.length;
        }
      });
      
      // Calculate entity box size based on content
      const width = Math.max(
        400,
        entity.name.length * 20,
        (maxAttributeLength * 14) + (maxTypeLength * 10) + 80
      );
      
      // Increase height calculation for entities with many attributes
      const height = Math.max(200, (entity.attributes.length + 1) * 40 + 60);
      
      
      // Adjust x and y to center the entity in its cell
      const x = col * cellWidth + (cellWidth - width) / 2;
      const y = row * cellHeight + (cellHeight - height) / 2;
      
      positions[entity.name] = { x, y, width, height };
    });
    
    return positions;
  }, [entityData]);
  
  // Handle zoom in/out
  const handleZoom = (zoomIn: boolean) => {
    setScale(prevScale => {
      const newScale = zoomIn ? prevScale * 1.2 : prevScale / 1.2;
      // Limit scale between 0.5 and 3
      return Math.min(Math.max(newScale, 0.5), 3);
    });
  };
  
  // Handle mouse down for dragging
  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return; // Only left mouse button
    setIsDragging(true);
    setDragStart({ x: e.clientX, y: e.clientY });
  };
  
  // Handle mouse move for dragging
  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    
    const dx = e.clientX - dragStart.x;
    const dy = e.clientY - dragStart.y;
    
    setPosition(prev => ({
      x: prev.x + dx,
      y: prev.y + dy
    }));
    
    setDragStart({ x: e.clientX, y: e.clientY });
  };
  
  // Handle mouse up to stop dragging
  const handleMouseUp = () => {
    setIsDragging(false);
  };
  
  // Handle reset zoom and position
  const handleResetZoom = () => {
    setScale(1.0);
    setPosition({ x: 0, y: 0 });
  };
  
  // Handle fullscreen toggle
  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen);
    // Reset position when toggling fullscreen
    setPosition({ x: 0, y: 0 });
    
    // Delay scale adjustment to allow fullscreen transition
    setTimeout(() => {
      if (containerRef.current && diagramRef.current) {
        const containerWidth = containerRef.current.clientWidth;
        const diagramWidth = cols * 700 + 160; // Estimate diagram width
        const newScale = Math.min(containerWidth / diagramWidth, 1.0);
        setScale(newScale);
      }
    }, 100);
  };
  
  // Calculate the number of columns for the grid
  const cols = Math.max(2, Math.ceil(Math.sqrt(entityData.entities.length * (16/9/3))));
  
  // Calculate relationships
  const relationships = useMemo(() => {
    return entityData.relationships.map(rel => {
      const fromPos = entityPositions[rel.from];
      const toPos = entityPositions[rel.to];
      
      if (!fromPos || !toPos) return null;
      
      // Calculate connection points
      const fromCenterX = fromPos.x + fromPos.width / 2;
      const fromCenterY = fromPos.y + fromPos.height / 2;
      const toCenterX = toPos.x + toPos.width / 2;
      const toCenterY = toPos.y + toPos.height / 2;
      
      // Calculate angle between entities
      const angle = Math.atan2(toCenterY - fromCenterY, toCenterX - fromCenterX);
      
      // Calculate connection points on the edges of the entities
      let fromX, fromY, toX, toY;
      
      // From entity connection point
      if (Math.abs(Math.cos(angle)) > Math.abs(Math.sin(angle))) {
        // Connect to left or right side
        fromX = fromCenterX + (Math.sign(Math.cos(angle)) * fromPos.width / 2);
        fromY = fromCenterY + Math.tan(angle) * (Math.sign(Math.cos(angle)) * fromPos.width / 2);
      } else {
        // Connect to top or bottom side
        fromY = fromCenterY + (Math.sign(Math.sin(angle)) * fromPos.height / 2);
        fromX = fromCenterX + (Math.sin(angle) !== 0 ? (Math.sign(Math.sin(angle)) * fromPos.height / 2) / Math.tan(angle) : 0);
      }
      
      // To entity connection point
      if (Math.abs(Math.cos(angle)) > Math.abs(Math.sin(angle))) {
        // Connect to left or right side
        toX = toCenterX - (Math.sign(Math.cos(angle)) * toPos.width / 2);
        toY = toCenterY - Math.tan(angle) * (Math.sign(Math.cos(angle)) * toPos.width / 2);
      } else {
        // Connect to top or bottom side
        toY = toCenterY - (Math.sign(Math.sin(angle)) * toPos.height / 2);
        toX = toCenterX - (Math.sin(angle) !== 0 ? (Math.sign(Math.sin(angle)) * toPos.height / 2) / Math.tan(angle) : 0);
      }
      
      return {
        from: { x: fromX, y: fromY },
        to: { x: toX, y: toY },
        type: rel.type
      };
    }).filter(Boolean);
  }, [entityData.relationships, entityPositions]);
  
  return (
    <Box sx={{ 
      display: 'flex', 
      flexDirection: 'column',
      width: '100%',
      height: isFullscreen ? '100vh' : '80vh',
      position: isFullscreen ? 'fixed' : 'relative',
      top: isFullscreen ? 0 : 'auto',
      left: isFullscreen ? 0 : 'auto',
      right: isFullscreen ? 0 : 'auto',
      bottom: isFullscreen ? 0 : 'auto',
      zIndex: isFullscreen ? 1300 : 1,
      bgcolor: 'background.paper',
      boxShadow: isFullscreen ? 24 : 0,
      p: 2
    }}>
      <Box sx={{ 
        display: 'flex', 
        justifyContent: 'space-between',
        mb: 2
      }}>
        <Typography variant="h6">
          Database Schema Visualization {isFullscreen ? '(Fullscreen Mode)' : ''}
        </Typography>
        <Box>
          <Button 
            onClick={() => handleZoom(true)} 
            variant="outlined" 
            size="small" 
            sx={{ minWidth: '40px', mr: 1 }}
          >
            +
          </Button>
          <Button 
            onClick={() => handleZoom(false)} 
            variant="outlined" 
            size="small" 
            sx={{ minWidth: '40px', mr: 1 }}
          >
            -
          </Button>
          <Button 
            onClick={handleResetZoom} 
            variant="outlined" 
            size="small" 
            sx={{ mr: 1 }}
          >
            Reset
          </Button>
          <Button 
            onClick={toggleFullscreen} 
            variant="outlined" 
            size="small"
          >
            {isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
          </Button>
        </Box>
      </Box>
      
      <Box 
        ref={containerRef}
        sx={{ 
          flex: 1, 
          overflow: 'hidden',
          border: '1px solid',
          borderColor: 'divider',
          borderRadius: 1,
          position: 'relative',
          bgcolor: '#f5f5f5'
        }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        <Box 
          ref={diagramRef}
          sx={{ 
            position: 'absolute',
            transform: `scale(${scale}) translate(${position.x}px, ${position.y}px)`,
            transformOrigin: '0 0',
            transition: 'transform 0.1s ease-out',
            cursor: isDragging ? 'grabbing' : 'grab'
          }}
        >
          {/* Draw relationships */}
          <svg 
            width={cols * 700 + 160} 
            height={(Math.ceil(entityData.entities.length / cols)) * 650 + 160} // Updated to match new cellHeight
            style={{ 
              position: 'absolute', 
              top: 0, 
              left: 0, 
              pointerEvents: 'none'
            }}
          >
            {relationships.map((rel, index) => rel && (
              <g key={index}>
                <line 
                  x1={rel.from.x} 
                  y1={rel.from.y} 
                  x2={rel.to.x} 
                  y2={rel.to.y} 
                  stroke={theme.palette.primary.main}
                  strokeWidth={2}
                  markerEnd="url(#arrowhead)"
                />
                {/* Relationship type indicator */}
                <text 
                  x={(rel.from.x + rel.to.x) / 2} 
                  y={(rel.from.y + rel.to.y) / 2 - 10}
                  textAnchor="middle"
                  fill={theme.palette.text.primary}
                  fontSize={14}
                  fontWeight="bold"
                  style={{ pointerEvents: 'none' }}
                >
                  {rel.type}
                </text>
              </g>
            ))}
            {/* Arrow marker definition */}
            <defs>
              <marker
                id="arrowhead"
                markerWidth="10"
                markerHeight="7"
                refX="9"
                refY="3.5"
                orient="auto"
              >
                <polygon 
                  points="0 0, 10 3.5, 0 7" 
                  fill={theme.palette.primary.main}
                />
              </marker>
            </defs>
          </svg>
          
          {/* Draw entities */}
          {entityData.entities.map((entity) => {
            const pos = entityPositions[entity.name];
            if (!pos) return null;
            
            return (
              <Box
                key={entity.name}
                sx={{
                  position: 'absolute',
                  left: pos.x,
                  top: pos.y,
                  width: pos.width,
                  height: pos.height,
                  bgcolor: 'background.paper',
                  borderRadius: 2,
                  boxShadow: 3,
                  overflow: 'hidden',
                  display: 'flex',
                  flexDirection: 'column',
                  border: '1px solid',
                  borderColor: 'divider'
                }}
              >
                {/* Entity header */}
                <Box sx={{ 
                  bgcolor: theme.palette.primary.main,
                  color: 'white',
                  p: 2,
                  textAlign: 'center',
                  fontWeight: 'bold',
                  fontSize: '1.2rem'
                }}>
                  {entity.name}
                </Box>
                
                {/* Entity attributes */}
                <Box sx={{ p: 2, flex: 1, overflow: 'auto' }}>
                  {entity.attributes.map((attr, index) => (
                    <Box 
                      key={index}
                      sx={{ 
                        display: 'flex', 
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        py: 0.75,
                        borderBottom: index < entity.attributes.length - 1 ? '1px solid' : 'none',
                        borderColor: 'divider'
                      }}
                    >
                      <Box sx={{ 
                        display: 'flex', 
                        alignItems: 'center',
                        width: '60%',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap'
                      }}>
                        {attr.key === 'PK' && (
                          <span role="img" aria-label="Primary Key" style={{ marginRight: '8px' }}>
                            🔑
                          </span>
                        )}
                        {attr.key === 'FK' && (
                          <span role="img" aria-label="Foreign Key" style={{ marginRight: '8px' }}>
                            🔗
                          </span>
                        )}
                        {attr.name}
                      </Box>
                      <Box sx={{ 
                        bgcolor: '#f0f0f0',
                        px: 1.5,
                        py: 0.5,
                        borderRadius: 1,
                        fontSize: '0.9rem',
                        fontFamily: 'monospace',
                        color: theme.palette.text.secondary,
                        maxWidth: '40%',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap'
                      }}>
                        {attr.type}
                      </Box>
                    </Box>
                  ))}
                </Box>
              </Box>
            );
          })}
        </Box>
      </Box>
      
      {/* Zoom indicator */}
      <Box sx={{ 
        position: 'absolute', 
        bottom: 16, 
        right: 16, 
        bgcolor: 'rgba(255,255,255,0.8)',
        px: 2,
        py: 0.5,
        borderRadius: 1,
        boxShadow: 1
      }}>
        Zoom: {Math.round(scale * 100)}%
      </Box>
    </Box>
  );
};

export default function DiscoverScreen() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [entityData, setEntityData] = useState<EntityRelationship | null>(null);
  const [,setConversationId] = useState<string | null>(null);
  const [,setLastMessage] = useState<Message | null>(null);
  const [,setIsComplete] = useState(false);
  const [showDetailedView, setShowDetailedView] = useState(false);
  const subscriptionRef = React.useRef<any>(null);

  // Function to create a conversation and send a message with action = 'get_entity_relationship'
  const fetchEntityRelationshipData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Create a new conversation
      const res = await client.graphql({
        query: createConversation,
      });
      
      const conversation = res.data?.createConversation?.conversation;
      
      if (!conversation?.conversationId) {
        throw new Error('Failed to create conversation: Invalid response');
      }
      
      const newConversationId = conversation.conversationId;
      setConversationId(newConversationId);
      
      // Send a message with action = 'get_entity_relationship'
      await client.graphql({
        query: createMessageAsync,
        variables: {
          input: {
            conversationId: newConversationId,
            prompt: JSON.stringify({
              action: 'get_entity_relationship'
            })
          }
        }
      });

      // Set up subscription to receive message chunks
      setupSubscription(newConversationId);
      
      // Fetch initial state
      await fetchInitialState(newConversationId);
      
    } catch (err) {
      console.error('Error fetching entity relationship data:', err);
      setError('Failed to fetch entity relationship data. Please try again.');
      setLoading(false);
    }
  };

  // Function to fetch initial state
  const fetchInitialState = async (id: string) => {
    try {
      console.log("[Initial State] Fetching for ID:", id);
      const result = await client.graphql({
        query: getConversation,
        variables: { input: { conversationId: id } }
      });

      if (!result.data) {
        throw new Error('No data returned from query');
      }

      console.log("[Initial State] Received:", result.data);
      const conversation = result.data.getConversation;
      console.log("[Initial State] Status:", conversation?.status);
      
      const messages = conversation?.messages || [];
      if (messages.length > 0) {
        const lastMsg = messages[messages.length - 1];
        console.log("[Initial State] Last message:", lastMsg);
        
        if (lastMsg.sender === 'Agent' || lastMsg.sender === 'Assistant') {
          console.log("[Initial State] Found complete Assistant message");
          setLastMessage(lastMsg);
          setLoading(false);
          setIsComplete(true);
          
          // Try to parse the message with our custom parser
          try {
            const data = parseEntityRelationshipData(lastMsg.message);
            setEntityData(data);
          } catch (e) {
            console.error("Error parsing entity relationship data:", e);
            setError("Failed to parse entity relationship data");
          }
        }
      }

      if (conversation?.status === 'COMPLETE') {
        console.log("[Initial State] Conversation is complete");
        setIsComplete(true);
      }
    } catch (err) {
      console.error("[Initial State] Error:", err);
      setError("Failed to load conversation");
      setLoading(false);
    }
  };

  // Function to set up subscription
  const setupSubscription = (id: string) => {
    try {
      console.log("[Subscription] Creating subscription object");
      
      // Clean up any existing subscription
      if (subscriptionRef.current) {
        console.log("[Subscription] Cleaning up existing subscription");
        subscriptionRef.current.unsubscribe();
        subscriptionRef.current = null;
      }
      
      const subscription = client
        .graphql({
          query: recieveMessageChunkAsync,
          variables: { input: { conversationId: id } }
        })
        .subscribe({
          next: (response) => {
            console.log("[Subscription] Message received:", response);
            const messageChunk = response.data?.recieveMessageChunkAsync;
            
            if (!messageChunk) {
              console.log("[Subscription] No message chunk in response");
              return;
            }

            console.log("[Subscription] Processing chunk:", {
              type: messageChunk.chunkType,
              status: messageChunk.status,
              content: messageChunk.chunk
            });

            if (messageChunk.chunkType === 'text') {
              console.log("[Subscription] Processing text chunk");
              setLastMessage((prevMessage) => {
                const updatedMessage = {
                  __typename: "Message" as const,
                  sender: 'Assistant',
                  message: prevMessage?.sender === 'Assistant' 
                    ? (prevMessage.message || '') + (messageChunk.chunk || '')
                    : messageChunk.chunk || '',
                  createdAt: prevMessage?.createdAt || new Date().toISOString()
                };
                console.log("[Subscription] Updated message:", updatedMessage);
                
                // Try to parse the message with our custom parser
                try {
                  const data = parseEntityRelationshipData(updatedMessage.message);
                  setEntityData(data);
                } catch (e) {
                  console.error("Error parsing entity relationship data:", e);
                  // Don't set error here as the message might be incomplete
                }
                
                return updatedMessage;
              });
              setLoading(false);
            }
            
            if (messageChunk.chunkType === 'error') {
              console.error("[Subscription] Error chunk received:", messageChunk.chunk);
              setError(messageChunk.chunk || "An error occurred");
              setLoading(false);
            }

            if (messageChunk.chunkType === 'status' && messageChunk.status === 'COMPLETE') {
              console.log("[Subscription] Processing complete");
              setLoading(false);
              setIsComplete(true);
            }
          },
          error: (error) => {
            console.error("[Subscription] Error in subscription:", error);
            setError("Error receiving updates");
            setLoading(false);
          },
          complete: () => {
            console.log("[Subscription] Subscription completed normally");
          }
        });

      subscriptionRef.current = subscription;
      console.log("[Subscription] Setup completed successfully");
    } catch (error) {
      console.error("[Subscription] Error during setup:", error);
      setError("Failed to setup message updates");
      setLoading(false);
    }
  };

  // Fetch entity relationship data when component mounts
  useEffect(() => {
    fetchEntityRelationshipData();
    
    // Clean up subscription when component unmounts
    return () => {
      if (subscriptionRef.current) {
        subscriptionRef.current.unsubscribe();
      }
    };
  }, []);

  const handleCreateConversation = async () => {
    try {
      const res = await client.graphql({
        query: createConversation,
      });
      
      const conversation = res.data?.createConversation?.conversation;
      
      if (!conversation?.conversationId) {
        throw new Error('Failed to create conversation: Invalid response');
      }
      
      await client.graphql({
        query: createMessageAsync,
        variables: {
          input: {
            conversationId: conversation.conversationId,
            prompt: JSON.stringify({
                action: 'discover'
              })
          }
        }
      });

      return conversation.conversationId;
    } catch (err) {
      console.error('Error creating conversation:', err);
      throw err;
    }
  };
  
  const handleProcess = async () => {
    const conversationId = await handleCreateConversation();

    navigate('/review', { 
        state: { 
            uploadId: conversationId,
            fromUpload: true
        },
        replace: true
    });
  };

  // Render entity relationship data
  const renderEntityData = () => {
    if (!entityData) return null;
    
    return (
      <Box sx={{ mt: 4 }}>
        <Paper 
          elevation={3}
          sx={{
            p: 4,
            backgroundColor: 'background.paper',
            borderRadius: 2,
            width: '100%' // Ensure paper uses full width
          }}
        >
          <Typography variant="h5" component="h2" gutterBottom>
            Database Entity Relationships
          </Typography>
          
          {/* Add the ER Diagram component */}
          <ERDiagram entityData={entityData} />
          
          {/* Toggle button to show/hide detailed entity list */}
          <Box sx={{ mt: 3, mb: 2, display: 'flex', justifyContent: 'center' }}>
            <Button 
              variant="outlined" 
              size="small"
              onClick={() => setShowDetailedView(!showDetailedView)}
            >
              {showDetailedView ? 'Hide Detailed View' : 'Show Detailed View'}
            </Button>
          </Box>
          
          {/* Detailed entity list (conditionally rendered) */}
          {showDetailedView && (
            <>
              {entityData.entities.length > 0 && (
                <Box sx={{ mb: 4 }}>
                  <Typography variant="h6" gutterBottom>
                    Entities
                  </Typography>
                  {entityData.entities.map((entity, index) => (
                    <Box key={index} sx={{ mb: 3, pl: 2, borderLeft: '2px solid', borderColor: 'primary.main' }}>
                      <Typography variant="subtitle1" fontWeight="bold">
                        {entity.name} {entity.type && `(${entity.type})`}
                      </Typography>
                      <Typography variant="body2" component="div" sx={{ mt: 1 }}>
                        Attributes:
                        <Box component="ul" sx={{ pl: 2, mt: 0.5 }}>
                          {entity.attributes.map((attr, attrIndex) => (
                            <Box component="li" key={attrIndex} sx={{ mb: 0.5 }}>
                              <Typography variant="body2">
                                <strong>{attr.name}</strong>: {attr.type}
                                {attr.key && <span style={{ color: attr.key === 'PK' ? '#2e7d32' : '#1976d2', marginLeft: '4px' }}>
                                  ({attr.key})
                                </span>}
                              </Typography>
                            </Box>
                          ))}
                        </Box>
                      </Typography>
                    </Box>
                  ))}
                </Box>
              )}
              
              {entityData.relationships.length > 0 && (
                <Box>
                  <Typography variant="h6" gutterBottom>
                    Relationships
                  </Typography>
                  {entityData.relationships.map((rel, index) => (
                    <Box key={index} sx={{ mb: 1.5, pl: 2, borderLeft: '2px solid', borderColor: 'secondary.main' }}>
                      <Typography variant="body1">
                        <strong>{rel.from}</strong> <span style={{ color: '#ff4081' }}>{rel.type}</span> <strong>{rel.to}</strong>
                      </Typography>
                      {(rel.fromAttribute || rel.toAttribute) && (
                        <Typography variant="body2" sx={{ mt: 0.5, color: 'text.secondary' }}>
                          {rel.fromAttribute && <span>via {rel.from}.{rel.fromAttribute}</span>}
                          {rel.fromAttribute && rel.toAttribute && <span> → </span>}
                          {rel.toAttribute && <span>{rel.to}.{rel.toAttribute}</span>}
                        </Typography>
                      )}
                    </Box>
                  ))}
                </Box>
              )}
            </>
          )}
        </Paper>
      </Box>
    );
  };

  return (
    <Container maxWidth={false} sx={{ px: { xs: 2, sm: 3, md: 4 } }}>
    <Box sx={{ mt: 4 }}>
        <Paper 
          elevation={3}
          sx={{
            p: 4,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            backgroundColor: 'background.paper',
            borderRadius: 2
          }}
        >
          <Typography variant="h4" component="h1" gutterBottom align="center">
            Discover AI/ML Use Cases
          </Typography>
          <Typography variant="body1" sx={{ mb: 4, textAlign: 'center', maxWidth: '600px' }}>
            Let AI help you discover potential machine learning use cases from your data.
          </Typography>
          <Button 
            variant="contained" 
            color="primary" 
            size="large"
            onClick={handleProcess}
            sx={{
              minWidth: '200px',
              height: '48px',
              fontSize: '1.1rem',
              textTransform: 'none',
              fontWeight: 500,
              boxShadow: 2,
              '&:hover': {
                boxShadow: 4,
              }
            }}
          >
            Discover Use Cases
          </Button>
        </Paper>
      </Box>
      
      {loading && (
        <Box sx={{ 
          display: 'flex', 
          justifyContent: 'center',
          alignItems: 'center',
          mt: 8,
          mb: 4
        }}>
          <CircularProgress size={40} />
        </Box>
      )}
      
      {error && (
        <Box sx={{ mt: 4 }}>
          <Paper sx={{ p: 3, bgcolor: '#ffebee' }}>
            <Typography color="error">
              {error}
            </Typography>
          </Paper>
        </Box>
      )}
      
      {/* Render entity relationship data if available */}
      {entityData && renderEntityData()}
      
      
    </Container>
  )
} 