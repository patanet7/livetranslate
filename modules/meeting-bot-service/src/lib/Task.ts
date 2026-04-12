import { Logger } from 'winston';

/**
 * Abstract base class for long-running tasks with logging.
 * @template TArgs - Input arguments type (use null if no input needed)
 * @template TResult - Return type (use void if no result needed)
 */
export abstract class Task<TArgs, TResult> {
  protected _logger: Logger;

  constructor(logger: Logger) {
    this._logger = logger;
  }

  /**
   * Execute the task with the given input.
   * Subclasses must implement this method.
   */
  protected abstract execute(input: TArgs): Promise<TResult>;

  /**
   * Run the task. Wrapper that could add pre/post processing.
   */
  async run(input: TArgs): Promise<TResult> {
    return this.execute(input);
  }

  /**
   * Alias for run() - maintains backwards compatibility.
   */
  async runAsync(input: TArgs): Promise<TResult> {
    return this.execute(input);
  }
}
