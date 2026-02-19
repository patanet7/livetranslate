# Multi-Platform Meeting Bot Architecture Plan
## LiveTranslate Phase 3.4: ScreenApp Integration

**Date**: 2025-10-28
**Status**: PLANNING
**Priority**: HIGH

---

## 🎯 Executive Summary

Integrate battle-tested multi-platform meeting bot architecture from ScreenApp.ai into LiveTranslate, enabling support for:
- ✅ **Google Meet** (improved with stealth + auto-accept)
- ✅ **Microsoft Teams** (new)
- ✅ **Zoom** (new)
- ✅ **Extensible plugin system** for future platforms

---

## 📊 Current State Analysis

### LiveTranslate Bot Container (Current)
**Location**: `modules/bot-container/`

**Current Implementation**:
```python
# Single class for Google Meet only
src/google_meet_automation.py
  - GoogleMeetAutomation class
  - BrowserConfig dataclass
  - Playwright with basic stealth
  - Login-based authentication (requires Google account)
```

**Limitations**:
1. ❌ Only supports Google Meet
2. ❌ Requires authentication (Google account)
3. ❌ Getting 401 errors / bot detection
4. ❌ Not extensible for other platforms
5. ❌ No plugin architecture
6. ❌ No API token support for callbacks

---

## 🏗️ ScreenApp Architecture Analysis

### Key Components to Port

#### 1. **Abstract Bot System** ⭐
```typescript
AbstractMeetBot (interface)
  ↓
MeetBotBase (shared functionality)
  ↓
├── GoogleMeetBot
├── MicrosoftTeamsBot
└── ZoomBot
```

**Benefits**:
- Factory pattern for bot creation
- Shared retry logic, error handling
- Consistent state management
- Easy to add new platforms

#### 2. **Advanced Stealth Configuration** ⭐⭐⭐
```typescript
// src/lib/chromium.ts
playwright-extra + puppeteer-extra-plugin-stealth
+ Critical browser args:
  - '--auto-accept-this-tab-capture'  // ← KEY FOR RECORDING
  - '--enable-usermedia-screen-capturing'
  - '--enable-features=MediaRecorder'
  - '--use-gl=angle'
  - '--use-angle=swiftshader'
  - Linux X11 user agent
```

**Why This Works**:
- Bypasses most bot detection
- Auto-accepts screen capture (no manual intervention)
- Tested in production with thousands of bots

#### 3. **In-Browser Recording** (Optional)
```typescript
// Uses MediaRecorder API in browser context
// Streams chunks back via page.exposeFunction()
// Supports VP9, WebM codecs
```

**Decision**: Keep LiveTranslate's current audio pipeline, but improve browser automation.

#### 4. **Lobby & Admission Handling** ⭐
```typescript
// Sophisticated waiting room detection
- Polls for "People" button, "Leave call" button
- Detects admission/denial messages
- Configurable timeout with retry
- Smart participant detection
```

#### 5. **API Integration** ⭐
```typescript
interface JoinParams {
  url: string
  name: string
  bearerToken: string     // ← For callbacks
  teamId: string
  userId: string
  botId?: string
  eventId?: string
}

// POST /google/join, /microsoft/join, /zoom/join
// Status callbacks to backend via bearerToken
```

#### 6. **Redis Queue Support** (Optional)
```typescript
// Async job processing
// RPUSH to queue → BLPOP consumer
// Single job execution model
// FIFO processing
```

---

## 🎨 Proposed Architecture for LiveTranslate

### Phase 1: Core Abstractions

#### File Structure
```
modules/bot-container/
├── src/
│   ├── bots/
│   │   ├── __init__.py
│   │   ├── abstract_bot.py           # Base interface
│   │   ├── base_bot.py               # Shared functionality
│   │   ├── google_meet_bot.py        # Google Meet implementation
│   │   ├── microsoft_teams_bot.py    # Teams implementation
│   │   ├── zoom_bot.py               # Zoom implementation
│   │   └── bot_factory.py            # Factory + registry
│   │
│   ├── browser/
│   │   ├── __init__.py
│   │   ├── launcher.py               # Browser launch with stealth
│   │   ├── stealth_config.py         # Stealth configuration
│   │   └── audio_capture.py          # Audio extraction (keep current)
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── bot_config.py             # Configuration models
│   │   ├── join_params.py            # Join parameters
│   │   └── bot_status.py             # Status tracking
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── callback_service.py       # Webhook/API callbacks
│   │   └── retry_service.py          # Retry logic
│   │
│   └── api/
│       ├── __init__.py
│       ├── routes.py                 # FastAPI endpoints
│       └── schemas.py                # Pydantic models
│
├── tests/
│   ├── test_google_meet_bot.py
│   ├── test_teams_bot.py
│   └── test_zoom_bot.py
│
└── requirements.txt                   # Updated dependencies
```

### Phase 2: Core Classes

#### 1. AbstractMeetingBot
```python
from abc import ABC, abstractmethod
from typing import Optional, Callable
from dataclasses import dataclass

@dataclass
class JoinParams:
    """Parameters for joining a meeting"""
    url: str
    name: str
    provider: str  # 'google' | 'microsoft' | 'zoom'

    # Optional authentication
    bearer_token: Optional[str] = None

    # Identifiers
    team_id: Optional[str] = None
    user_id: Optional[str] = None
    bot_id: Optional[str] = None
    event_id: Optional[str] = None

    # Callbacks
    status_callback_url: Optional[str] = None
    webhook_url: Optional[str] = None

    # Configuration
    timezone: str = "UTC"
    max_duration_minutes: int = 180
    audio_output_path: Optional[str] = None

class BotStatus(Enum):
    """Bot lifecycle states"""
    INITIALIZING = "initializing"
    CONNECTING = "connecting"
    JOINING = "joining"
    WAITING_ROOM = "waiting_room"
    JOINED = "joined"
    ACTIVE = "active"
    RECORDING = "recording"
    LEAVING = "leaving"
    FINISHED = "finished"
    FAILED = "failed"
    ERROR = "error"

class AbstractMeetingBot(ABC):
    """Base interface for all meeting bot implementations"""

    def __init__(self, logger: Logger):
        self.logger = logger
        self.page: Optional[Page] = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self._status: BotStatus = BotStatus.INITIALIZING
        self._status_history: List[BotStatus] = []

    @abstractmethod
    async def join(self, params: JoinParams) -> bool:
        """Join a meeting and start processing"""
        pass

    @abstractmethod
    async def leave(self) -> None:
        """Leave the meeting and cleanup"""
        pass

    @abstractmethod
    def get_platform_name(self) -> str:
        """Return the platform identifier (e.g., 'google', 'microsoft', 'zoom')"""
        pass

    # Shared helper methods
    async def update_status(self, status: BotStatus, callback_fn: Optional[Callable] = None):
        """Update bot status and trigger callbacks"""
        self._status = status
        self._status_history.append(status)

        if callback_fn:
            await callback_fn(status)

    async def cleanup(self):
        """Cleanup browser resources"""
        if self.page:
            await self.page.close()
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
```

#### 2. BotFactory with Registry
```python
class BotRegistry:
    """Registry for bot implementations"""

    _bots: Dict[str, Type[AbstractMeetingBot]] = {}

    @classmethod
    def register(cls, provider: str):
        """Decorator to register bot implementations"""
        def decorator(bot_class: Type[AbstractMeetingBot]):
            cls._bots[provider.lower()] = bot_class
            return bot_class
        return decorator

    @classmethod
    def get_bot_class(cls, provider: str) -> Type[AbstractMeetingBot]:
        """Get bot class for provider"""
        bot_class = cls._bots.get(provider.lower())
        if not bot_class:
            raise ValueError(f"No bot registered for provider: {provider}")
        return bot_class

    @classmethod
    def list_providers(cls) -> List[str]:
        """List all registered providers"""
        return list(cls._bots.keys())

class BotFactory:
    """Factory for creating bot instances"""

    @staticmethod
    def create_bot(params: JoinParams, logger: Logger) -> AbstractMeetingBot:
        """Create appropriate bot instance based on provider"""
        bot_class = BotRegistry.get_bot_class(params.provider)
        return bot_class(logger)

    @staticmethod
    def detect_provider_from_url(url: str) -> str:
        """Auto-detect provider from meeting URL"""
        if "meet.google.com" in url:
            return "google"
        elif "teams.microsoft.com" in url:
            return "microsoft"
        elif "zoom.us" in url or "zoom.com" in url:
            return "zoom"
        else:
            raise ValueError(f"Could not detect meeting provider from URL: {url}")
```

#### 3. Enhanced Browser Launcher
```python
from playwright.async_api import async_playwright, Browser
from playwright_stealth import stealth_async

class StealthBrowserLauncher:
    """Launch Playwright browser with ScreenApp's battle-tested stealth config"""

    @staticmethod
    async def launch(url: str, headless: bool = True) -> Page:
        """
        Launch browser with stealth configuration from ScreenApp

        Key features:
        - Auto-accepts screen capture
        - Stealth plugin to hide automation
        - Linux X11 user agent
        - GL acceleration via ANGLE/SwiftShader
        """

        browser_args = [
            # ScreenApp's critical args for recording
            '--enable-usermedia-screen-capturing',
            '--allow-http-screen-capture',
            '--auto-accept-this-tab-capture',  # ⭐ Auto-accept screen sharing
            '--enable-features=MediaRecorder',

            # Security bypasses
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-web-security',

            # Graphics acceleration
            '--use-gl=angle',
            '--use-angle=swiftshader',

            # Window config
            '--window-size=1280,720',
        ]

        playwright = await async_playwright().start()

        browser = await playwright.chromium.launch(
            headless=headless,
            args=browser_args,
            ignore_default_args=['--mute-audio'],  # Don't mute audio!
        )

        # Linux X11 user agent (ScreenApp uses this)
        user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36'

        context = await browser.new_context(
            viewport={'width': 1280, 'height': 720},
            user_agent=user_agent,
            ignore_https_errors=True,
        )

        # Grant permissions
        await context.grant_permissions(['microphone', 'camera'], origin=url)

        page = await context.new_page()

        # Apply stealth (hides navigator.webdriver, etc.)
        await stealth_async(page)

        return page
```

#### 4. GoogleMeetBot (Enhanced)
```python
@BotRegistry.register("google")
class GoogleMeetBot(AbstractMeetingBot):
    """Google Meet bot with ScreenApp improvements"""

    def get_platform_name(self) -> str:
        return "google"

    async def join(self, params: JoinParams) -> bool:
        """Join Google Meet with improved reliability"""

        try:
            # Launch browser with stealth
            self.page = await StealthBrowserLauncher.launch(params.url, headless=False)

            await self.update_status(BotStatus.CONNECTING)

            # Navigate to meeting
            await self.page.goto(params.url, wait_until='networkidle')
            await asyncio.sleep(2)  # Reduced from 5s

            # Dismiss device permission modal
            await self._dismiss_device_check()

            # Check for sign-in requirement (reject if needed)
            if await self._requires_signin():
                self.logger.error("Meeting requires sign-in - not supported")
                raise UnsupportedMeetingError("Meeting requires authentication")

            # Enter name
            await self._enter_name(params.name)

            # Click join button
            await self._click_join_button()

            await self.update_status(BotStatus.JOINING)

            # Wait for admission
            admitted = await self._wait_for_admission(timeout=params.max_duration_minutes)

            if not admitted:
                raise WaitingAtLobbyError("Failed to be admitted to meeting")

            await self.update_status(BotStatus.ACTIVE)

            # Start audio capture (use existing LiveTranslate pipeline)
            # ... existing audio capture code ...

            return True

        except Exception as e:
            await self.update_status(BotStatus.FAILED)
            raise

    async def _dismiss_device_check(self):
        """Dismiss 'Continue without microphone and camera' modal"""
        try:
            button = self.page.get_by_role('button', name='Continue without microphone and camera')
            await button.wait_for(timeout=30000)
            await button.click()
        except:
            self.logger.info("Device check modal not found")

    async def _requires_signin(self) -> bool:
        """Check if meeting requires Google sign-in"""
        url = await self.page.url()
        if url.startswith('https://accounts.google.com/'):
            return True

        signin_heading = await self.page.locator('h1', has_text='Sign in')
        return await signin_heading.count() > 0

    async def _click_join_button(self):
        """Click join button with multiple fallback selectors"""
        possible_texts = ['Ask to join', 'Join now', 'Join anyway']

        for text in possible_texts:
            try:
                button = self.page.locator('button', has_text=re.compile(text, re.I)).first()
                if await button.count() > 0:
                    await button.click(timeout=5000)
                    self.logger.info(f"Clicked join using '{text}'")
                    return
            except:
                continue

        raise RuntimeError("Could not find join button")

    async def _wait_for_admission(self, timeout: int = 120) -> bool:
        """Wait for bot to be admitted to meeting (ScreenApp's approach)"""
        end_time = asyncio.get_event_loop().time() + (timeout * 60)

        admission_indicators = [
            'button[aria-label="People"]',
            'button[aria-label*="Leave call"]',
        ]

        lobby_texts = [
            "You'll join the call when someone lets you in",
            "Asking to be let in",
        ]

        while asyncio.get_event_loop().time() < end_time:
            # Check for admission
            for selector in admission_indicators:
                try:
                    element = await self.page.query_selector(selector)
                    if element:
                        self.logger.info("Admitted to meeting!")
                        return True
                except:
                    pass

            # Check for lobby
            page_text = await self.page.evaluate("() => document.body.innerText")
            in_lobby = any(text in page_text for text in lobby_texts)

            if in_lobby:
                self.logger.info("Waiting in lobby...")

            # Check for denial
            if "You can't join this video call" in page_text:
                self.logger.error("Access denied by meeting settings")
                return False

            await asyncio.sleep(2)

        self.logger.warning("Timeout waiting for admission")
        return False
```

---

## 📋 Implementation Phases

### Phase 1: Foundation (Week 1)
- [ ] Create new file structure
- [ ] Implement AbstractMeetingBot
- [ ] Implement BotRegistry + BotFactory
- [ ] Update browser launcher with stealth config
- [ ] Port GoogleMeetBot with improvements

### Phase 2: Multi-Platform (Week 2)
- [ ] Implement MicrosoftTeamsBot
- [ ] Implement ZoomBot
- [ ] Test all platforms

### Phase 3: API Integration (Week 3)
- [ ] Create FastAPI endpoints (POST /join/{provider})
- [ ] Add bearer token support
- [ ] Implement status callbacks
- [ ] Add webhook support
- [ ] Optional: Redis queue integration

### Phase 4: Orchestration Integration (Week 4)
- [ ] Update orchestration service to use new bot architecture
- [ ] Update database schema for multi-platform
- [ ] Migrate existing Google Meet bots
- [ ] End-to-end testing

### Phase 5: Documentation & Testing (Week 5)
- [ ] Comprehensive tests for all platforms
- [ ] API documentation
- [ ] Migration guide
- [ ] Performance benchmarks

---

## 🔧 API Design

### REST API Endpoints

#### 1. **Join Meeting** (Generic)
```http
POST /api/v1/bot/join
Content-Type: application/json
Authorization: Bearer {token}

{
  "url": "https://meet.google.com/abc-defg-hij",
  "name": "LiveTranslate Bot",
  "provider": "google",  // Optional: auto-detect from URL

  // Optional identifiers
  "team_id": "team123",
  "user_id": "user456",
  "bot_id": "bot789",
  "event_id": "event101",

  // Optional callbacks
  "status_callback_url": "https://api.example.com/bot/status",
  "webhook_url": "https://api.example.com/webhooks/bot",

  // Optional config
  "timezone": "America/New_York",
  "max_duration_minutes": 120
}

Response 202 Accepted:
{
  "success": true,
  "bot_id": "bot789",
  "provider": "google",
  "status": "connecting",
  "message": "Bot is joining meeting"
}
```

#### 2. **Platform-Specific Endpoints** (ScreenApp compatible)
```http
POST /api/v1/bot/google/join
POST /api/v1/bot/microsoft/join
POST /api/v1/bot/zoom/join
```

#### 3. **Status Check**
```http
GET /api/v1/bot/{bot_id}/status

Response:
{
  "bot_id": "bot789",
  "provider": "google",
  "current_status": "active",
  "status_history": ["initializing", "connecting", "joining", "active"],
  "meeting_url": "https://meet.google.com/abc-defg-hij",
  "joined_at": "2025-10-28T10:30:00Z",
  "duration_seconds": 120
}
```

#### 4. **Leave Meeting**
```http
POST /api/v1/bot/{bot_id}/leave

Response:
{
  "success": true,
  "bot_id": "bot789",
  "status": "leaving"
}
```

#### 5. **List Supported Providers**
```http
GET /api/v1/bot/providers

Response:
{
  "providers": ["google", "microsoft", "zoom"],
  "total": 3
}
```

---

## 🔌 Extensibility: Adding New Platforms

### Plugin Pattern Example
```python
# To add a new platform (e.g., Webex):

@BotRegistry.register("webex")
class WebexBot(AbstractMeetingBot):
    """Cisco Webex Meetings bot"""

    def get_platform_name(self) -> str:
        return "webex"

    async def join(self, params: JoinParams) -> bool:
        # Platform-specific join logic
        self.page = await StealthBrowserLauncher.launch(params.url)

        # Navigate and join
        await self.page.goto(params.url)
        await self._webex_specific_join_flow()

        return True

    async def leave(self) -> None:
        # Platform-specific leave logic
        await self.cleanup()
```

**That's it!** The bot is automatically:
- ✅ Registered in the factory
- ✅ Available via API: `POST /api/v1/bot/webex/join`
- ✅ Accessible in orchestration service
- ✅ Listed in `GET /api/v1/bot/providers`

---

## 📦 Dependencies to Add

```txt
# requirements.txt additions

# Stealth mode
playwright-stealth==0.2.0

# Enhanced retry logic
tenacity==8.2.3

# Redis (optional, for queue support)
redis==5.0.1
aioredis==2.0.1

# HTTP callbacks
httpx==0.27.0

# Webhook validation
cryptography==41.0.7
```

---

## 🗄️ Database Schema Updates

```sql
-- Add provider column to bot_sessions
ALTER TABLE bot_sessions
ADD COLUMN provider VARCHAR(50) DEFAULT 'google';

-- Add index for provider lookups
CREATE INDEX idx_bot_sessions_provider ON bot_sessions(provider);

-- Add supported_providers table for dynamic registry
CREATE TABLE supported_providers (
    id SERIAL PRIMARY KEY,
    provider_name VARCHAR(50) UNIQUE NOT NULL,
    display_name VARCHAR(100) NOT NULL,
    is_enabled BOOLEAN DEFAULT TRUE,
    config JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO supported_providers (provider_name, display_name) VALUES
    ('google', 'Google Meet'),
    ('microsoft', 'Microsoft Teams'),
    ('zoom', 'Zoom');
```

---

## 🧪 Testing Strategy

### Unit Tests
```python
# tests/test_bot_factory.py
def test_factory_creates_google_bot():
    params = JoinParams(url="https://meet.google.com/test", provider="google")
    bot = BotFactory.create_bot(params, logger)
    assert isinstance(bot, GoogleMeetBot)

def test_factory_auto_detects_provider():
    provider = BotFactory.detect_provider_from_url("https://teams.microsoft.com/...")
    assert provider == "microsoft"
```

### Integration Tests
```python
# tests/test_google_meet_bot.py
@pytest.mark.integration
async def test_google_meet_join_public_meeting():
    """Test joining a public Google Meet"""
    params = JoinParams(
        url=os.getenv("TEST_GOOGLE_MEET_URL"),
        name="Test Bot",
        provider="google"
    )

    bot = BotFactory.create_bot(params, logger)
    success = await bot.join(params)

    assert success
    assert bot._status == BotStatus.ACTIVE

    await bot.leave()
```

---

## 🚀 Migration Path

### Step 1: Parallel Implementation
- Keep existing `google_meet_automation.py` working
- Implement new architecture alongside
- Gradual migration of features

### Step 2: Feature Flag
```python
# config.py
USE_NEW_BOT_ARCHITECTURE = os.getenv("USE_NEW_BOT_ARCHITECTURE", "false") == "true"

# orchestration service
if USE_NEW_BOT_ARCHITECTURE:
    bot = BotFactory.create_bot(params, logger)
else:
    bot = GoogleMeetAutomation(config)  # Legacy
```

### Step 3: Full Migration
- Switch all bots to new architecture
- Remove legacy code
- Update documentation

---

## 📈 Success Metrics

### Performance
- ✅ Join success rate > 95%
- ✅ Average join time < 30 seconds
- ✅ Bot detection rate < 5%

### Coverage
- ✅ Support for 3+ platforms (Google, Teams, Zoom)
- ✅ 100% test coverage for core bot classes
- ✅ API response time < 100ms

### Reliability
- ✅ Automatic retry on transient failures
- ✅ Graceful degradation on platform changes
- ✅ Comprehensive error logging

---

## 🎯 Next Steps

1. **Review this plan** - Get approval on architecture
2. **Set up dev environment** - Install dependencies, test ScreenApp bot
3. **Start Phase 1** - Create abstract base classes
4. **Port GoogleMeetBot** - With ScreenApp improvements
5. **Add Teams + Zoom** - Multi-platform support
6. **Integration** - Connect to orchestration service

---

## 📚 References

- **ScreenApp Meeting Bot**: https://github.com/screenappai/meeting-bot
- **Playwright Stealth**: https://github.com/AtuboDad/playwright_stealth
- **Playwright Extra**: https://github.com/Mattwmaster58/playwright_extra
- **Current LiveTranslate Bot**: `modules/bot-container/src/google_meet_automation.py`

---

**Author**: Claude Code
**Last Updated**: 2025-10-28
**Version**: 1.0-DRAFT
