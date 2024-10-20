import sys
import shutil
import os

from Context import Context
from Trainer import Trainer
from Merger import Merger
from Validator import Validator

def main():
    if not len(sys.argv) == 2:
        print("Argument 1 is missing: Configuration file")
        sys.exit(1)
    
    context = Context()
    context.load(sys.argv[1])
    
    purgeTargetDirectories(context)

    trainer = Trainer()
    trainer.train(context)
    
    validator = Validator();
    if not validator.validate(context) and context.vAbortOnFail:
        print("Validation not passed, aborting")
        sys.exit(0)
    
    merger = Merger()
    merger.mergeAndStore(context)
    
    print("Done")

def purgeTargetDirectories(context: Context):
    if not context.purgeTargetDirectories:
        return
    
    print("Purging target directories")
    for path in (context.locWorkdir, context.locAdapter, context.locFull):
        shutil.rmtree(path)
        os.mkdir(path)

if __name__ == "__main__":
    main()
