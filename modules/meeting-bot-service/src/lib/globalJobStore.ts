import { Logger } from 'winston';

interface JobResult {
  accepted: boolean;
}

/**
 * Global job store to ensure only one bot session runs at a time.
 * This prevents resource conflicts when multiple meeting join requests arrive.
 */
class GlobalJobStore {
  private _isBusy: boolean = false;
  private _currentJob: Promise<void> | null = null;
  private _shutdownRequested: boolean = false;

  /**
   * Check if a job is currently running.
   */
  isBusy(): boolean {
    return this._isBusy;
  }

  /**
   * Attempt to add a job to the store.
   * If the store is busy, the job is rejected.
   * If accepted, the job runs and the store becomes busy until completion.
   *
   * @param jobFn - Async function to execute
   * @param logger - Logger for the job
   * @returns JobResult indicating if the job was accepted
   */
  async addJob(jobFn: () => Promise<void>, logger: Logger): Promise<JobResult> {
    if (this._shutdownRequested) {
      logger.warn('JobStore shutdown requested, rejecting new job');
      return { accepted: false };
    }

    if (this._isBusy) {
      logger.warn('JobStore is busy, rejecting new job');
      return { accepted: false };
    }

    this._isBusy = true;
    logger.info('JobStore accepted new job');

    // Run the job asynchronously
    this._currentJob = (async () => {
      try {
        await jobFn();
      } catch (error) {
        logger.error('Job failed with error:', error);
      } finally {
        this._isBusy = false;
        this._currentJob = null;
        logger.info('JobStore released, ready for next job');
      }
    })();

    return { accepted: true };
  }

  /**
   * Wait for the current job to complete (if any).
   */
  async waitForCompletion(): Promise<void> {
    if (this._currentJob) {
      await this._currentJob;
    }
  }

  /**
   * Request shutdown - prevents new jobs from being accepted.
   */
  requestShutdown(): void {
    this._shutdownRequested = true;
  }

  /**
   * Check if shutdown has been requested.
   */
  isShutdownRequested(): boolean {
    return this._shutdownRequested;
  }
}

// Singleton instance
export const globalJobStore = new GlobalJobStore();
