# syntax=docker/dockerfile:1.4

# =============================================================================
# Stage 1: Dependencies
# Install and cache node dependencies
# =============================================================================
FROM node:25-alpine AS deps

# Labels for metadata
LABEL org.opencontainers.image.title="Frontend Service - Dependencies" \
      org.opencontainers.image.description="LiveTranslate frontend dependency stage"

# Enable pnpm
ENV PNPM_HOME=/pnpm
ENV PATH=$PNPM_HOME/bin:$PATH
RUN corepack enable

# Set working directory
WORKDIR /app

# Copy package files
COPY package.json pnpm-lock.yaml ./

# Install dependencies with frozen lockfile
RUN pnpm install --frozen-lockfile

# =============================================================================
# Stage 2: Build
# Build the production application
# =============================================================================
FROM deps AS build

WORKDIR /app

# Copy source files
COPY . .

# Build the application
RUN pnpm build

# =============================================================================
# Stage 3: Runtime
# Minimal production runtime with nginx
# =============================================================================
FROM nginx:1.29-alpine AS runtime

# Labels for metadata
LABEL org.opencontainers.image.title="Frontend Service" \
      org.opencontainers.image.description="Modern React UI for LiveTranslate" \
      org.opencontainers.image.vendor="LiveTranslate" \
      org.opencontainers.image.version="1.0.0"

# Install curl for health checks
RUN apk add --no-cache curl

# Create non-root user for nginx
RUN addgroup -g 1000 appuser && \
    adduser -u 1000 -G appuser -D -s /sbin/nologin appuser

# Copy built application
COPY --from=build /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Create nginx cache and pid directories with proper permissions
RUN mkdir -p /var/cache/nginx /var/run/nginx && \
    chown -R appuser:appuser /var/cache/nginx /var/run/nginx /usr/share/nginx/html && \
    chmod -R 755 /var/cache/nginx /var/run/nginx

# Configure nginx to run as non-root
RUN sed -i 's/user  nginx;/user  appuser;/' /etc/nginx/nginx.conf && \
    sed -i 's,/var/run/nginx.pid,/var/run/nginx/nginx.pid,' /etc/nginx/nginx.conf

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:80/ || exit 1

# Run nginx
CMD ["nginx", "-g", "daemon off;"]
