from gym.envs.registration import register

register(id='multi-coverage-v0',
         entry_point='multi_coverage.envs:MultiCoverageEnv')
