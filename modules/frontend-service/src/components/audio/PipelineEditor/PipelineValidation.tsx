import React from "react";
import {
  Box,
  Card,
  CardContent,
  Typography,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Collapse,
  IconButton,
  Chip,
  Divider,
  Alert,
} from "@mui/material";
import {
  Error,
  Warning,
  Info,
  ExpandMore,
  ExpandLess,
  CheckCircle,
  ReportProblem,
  Lightbulb,
} from "@mui/icons-material";

interface PipelineValidationProps {
  validationResult: {
    valid: boolean;
    errors: string[];
    warnings: string[];
    suggestions: string[];
  };
  compact?: boolean;
  onFix?: (issue: string) => void;
}

const PipelineValidation: React.FC<PipelineValidationProps> = ({
  validationResult,
  compact = false,
  onFix,
}) => {
  const [expandedSections, setExpandedSections] = React.useState({
    errors: true,
    warnings: true,
    suggestions: false,
  });

  const toggleSection = (section: keyof typeof expandedSections) => {
    setExpandedSections((prev) => ({
      ...prev,
      [section]: !prev[section],
    }));
  };

  if (compact) {
    return (
      <Alert
        severity={validationResult.valid ? "success" : "error"}
        icon={validationResult.valid ? <CheckCircle /> : <ReportProblem />}
        sx={{ py: 0.5 }}
      >
        <Typography variant="caption">
          {validationResult.valid ? (
            "Pipeline is valid and ready to process"
          ) : (
            <>
              {validationResult.errors.length} error
              {validationResult.errors.length !== 1 ? "s" : ""},{" "}
              {validationResult.warnings.length} warning
              {validationResult.warnings.length !== 1 ? "s" : ""}
            </>
          )}
        </Typography>
      </Alert>
    );
  }

  return (
    <Card sx={{ width: "100%", maxHeight: 400, overflow: "auto" }}>
      <CardContent sx={{ p: 2 }}>
        <Box
          display="flex"
          alignItems="center"
          justifyContent="space-between"
          mb={2}
        >
          <Typography variant="h6" component="h3">
            Pipeline Validation
          </Typography>
          <Box display="flex" gap={1}>
            {validationResult.errors.length > 0 && (
              <Chip
                size="small"
                label={`${validationResult.errors.length} Errors`}
                color="error"
                icon={<Error />}
              />
            )}
            {validationResult.warnings.length > 0 && (
              <Chip
                size="small"
                label={`${validationResult.warnings.length} Warnings`}
                color="warning"
                icon={<Warning />}
              />
            )}
            {validationResult.suggestions.length > 0 && (
              <Chip
                size="small"
                label={`${validationResult.suggestions.length} Tips`}
                color="info"
                icon={<Info />}
              />
            )}
          </Box>
        </Box>

        {/* Errors Section */}
        {validationResult.errors.length > 0 && (
          <Box mb={2}>
            <Box
              display="flex"
              alignItems="center"
              justifyContent="space-between"
              sx={{
                cursor: "pointer",
                p: 1,
                borderRadius: 1,
                "&:hover": { bgcolor: "action.hover" },
              }}
              onClick={() => toggleSection("errors")}
            >
              <Box display="flex" alignItems="center" gap={1}>
                <Error color="error" />
                <Typography variant="subtitle2" fontWeight="bold">
                  Errors ({validationResult.errors.length})
                </Typography>
              </Box>
              <IconButton size="small">
                {expandedSections.errors ? <ExpandLess /> : <ExpandMore />}
              </IconButton>
            </Box>

            <Collapse in={expandedSections.errors}>
              <List dense sx={{ pl: 2 }}>
                {validationResult.errors.map((error, index) => (
                  <ListItem
                    key={`error-${index}`}
                    sx={{
                      bgcolor: "error.light",
                      borderRadius: 1,
                      mb: 0.5,
                      py: 0.5,
                    }}
                    secondaryAction={
                      onFix && (
                        <Chip
                          size="small"
                          label="Fix"
                          onClick={() => onFix(error)}
                          sx={{
                            cursor: "pointer",
                            bgcolor: "error.main",
                            color: "white",
                            "&:hover": {
                              bgcolor: "error.dark",
                            },
                          }}
                        />
                      )
                    }
                  >
                    <ListItemIcon sx={{ minWidth: 32 }}>
                      <Error sx={{ fontSize: 16, color: "error.main" }} />
                    </ListItemIcon>
                    <ListItemText
                      primary={
                        <Typography variant="caption" color="error.dark">
                          {error}
                        </Typography>
                      }
                    />
                  </ListItem>
                ))}
              </List>
            </Collapse>
          </Box>
        )}

        {/* Warnings Section */}
        {validationResult.warnings.length > 0 && (
          <Box mb={2}>
            <Box
              display="flex"
              alignItems="center"
              justifyContent="space-between"
              sx={{
                cursor: "pointer",
                p: 1,
                borderRadius: 1,
                "&:hover": { bgcolor: "action.hover" },
              }}
              onClick={() => toggleSection("warnings")}
            >
              <Box display="flex" alignItems="center" gap={1}>
                <Warning color="warning" />
                <Typography variant="subtitle2" fontWeight="bold">
                  Warnings ({validationResult.warnings.length})
                </Typography>
              </Box>
              <IconButton size="small">
                {expandedSections.warnings ? <ExpandLess /> : <ExpandMore />}
              </IconButton>
            </Box>

            <Collapse in={expandedSections.warnings}>
              <List dense sx={{ pl: 2 }}>
                {validationResult.warnings.map((warning, index) => (
                  <ListItem
                    key={`warning-${index}`}
                    sx={{
                      bgcolor: "warning.light",
                      borderRadius: 1,
                      mb: 0.5,
                      py: 0.5,
                    }}
                  >
                    <ListItemIcon sx={{ minWidth: 32 }}>
                      <Warning sx={{ fontSize: 16, color: "warning.main" }} />
                    </ListItemIcon>
                    <ListItemText
                      primary={
                        <Typography variant="caption" color="warning.dark">
                          {warning}
                        </Typography>
                      }
                    />
                  </ListItem>
                ))}
              </List>
            </Collapse>
          </Box>
        )}

        {/* Suggestions Section */}
        {validationResult.suggestions.length > 0 && (
          <Box>
            <Box
              display="flex"
              alignItems="center"
              justifyContent="space-between"
              sx={{
                cursor: "pointer",
                p: 1,
                borderRadius: 1,
                "&:hover": { bgcolor: "action.hover" },
              }}
              onClick={() => toggleSection("suggestions")}
            >
              <Box display="flex" alignItems="center" gap={1}>
                <Lightbulb color="info" />
                <Typography variant="subtitle2" fontWeight="bold">
                  Suggestions ({validationResult.suggestions.length})
                </Typography>
              </Box>
              <IconButton size="small">
                {expandedSections.suggestions ? <ExpandLess /> : <ExpandMore />}
              </IconButton>
            </Box>

            <Collapse in={expandedSections.suggestions}>
              <List dense sx={{ pl: 2 }}>
                {validationResult.suggestions.map((suggestion, index) => (
                  <ListItem
                    key={`suggestion-${index}`}
                    sx={{
                      bgcolor: "info.light",
                      borderRadius: 1,
                      mb: 0.5,
                      py: 0.5,
                    }}
                  >
                    <ListItemIcon sx={{ minWidth: 32 }}>
                      <Lightbulb sx={{ fontSize: 16, color: "info.main" }} />
                    </ListItemIcon>
                    <ListItemText
                      primary={
                        <Typography variant="caption" color="info.dark">
                          {suggestion}
                        </Typography>
                      }
                    />
                  </ListItem>
                ))}
              </List>
            </Collapse>
          </Box>
        )}

        {/* Success Message */}
        {validationResult.valid && validationResult.errors.length === 0 && (
          <Alert severity="success" icon={<CheckCircle />}>
            <Typography variant="body2">
              Pipeline is valid and ready for audio processing!
            </Typography>
          </Alert>
        )}

        {/* Pipeline Requirements Info */}
        <Divider sx={{ my: 2 }} />
        <Box>
          <Typography variant="caption" color="text.secondary" gutterBottom>
            Pipeline Requirements:
          </Typography>
          <List dense>
            <ListItem sx={{ py: 0, pl: 1 }}>
              <ListItemIcon sx={{ minWidth: 24 }}>
                <CheckCircle sx={{ fontSize: 14, color: "success.main" }} />
              </ListItemIcon>
              <ListItemText
                primary={
                  <Typography variant="caption">
                    At least one input component (File or Microphone)
                  </Typography>
                }
              />
            </ListItem>
            <ListItem sx={{ py: 0, pl: 1 }}>
              <ListItemIcon sx={{ minWidth: 24 }}>
                <CheckCircle sx={{ fontSize: 14, color: "success.main" }} />
              </ListItemIcon>
              <ListItemText
                primary={
                  <Typography variant="caption">
                    At least one output component (Speaker or File)
                  </Typography>
                }
              />
            </ListItem>
            <ListItem sx={{ py: 0, pl: 1 }}>
              <ListItemIcon sx={{ minWidth: 24 }}>
                <CheckCircle sx={{ fontSize: 14, color: "success.main" }} />
              </ListItemIcon>
              <ListItemText
                primary={
                  <Typography variant="caption">
                    All components properly connected
                  </Typography>
                }
              />
            </ListItem>
            <ListItem sx={{ py: 0, pl: 1 }}>
              <ListItemIcon sx={{ minWidth: 24 }}>
                <CheckCircle sx={{ fontSize: 14, color: "success.main" }} />
              </ListItemIcon>
              <ListItemText
                primary={
                  <Typography variant="caption">
                    No circular connections (acyclic pipeline)
                  </Typography>
                }
              />
            </ListItem>
          </List>
        </Box>
      </CardContent>
    </Card>
  );
};

export default PipelineValidation;
